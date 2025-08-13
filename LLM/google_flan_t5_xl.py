from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn.functional as F
import time

# Load model
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Fixed location & object mappings
location_list = ["kitchen", "bathroom", "bedroom", "living room", "study room"]

object2locations = {
    "bed": ["bedroom"], "sofa": ["living room"], "stove": ["kitchen"], "counter": ["kitchen"],
    "microwave": ["kitchen"], "refrigerator": ["kitchen"], "toilet": ["bathroom"],
    "desktop_computer": ["study room"], "table": ["living room", "study room"],
    "sink": ["bathroom", "kitchen"], "window": ["kitchen", "bedroom"],
    "shelving": ["study room", "kitchen"], "curtain": ["bedroom", "bathroom"],
    "chair": ["living room", "study room"], "tv_monitor": ["living room", "bedroom", "study room"],
    "cabinet": ["study room", "kitchen", "bedroom"]
}

def normalize_room_name_llm(predicted_room: str) -> str:
    if predicted_room in location_list:
        return predicted_room

    prompt = (
        f"Map the following room name to the most appropriate one from this list:\n"
        f"{location_list}\n"
        f"Room name: '{predicted_room}'\n"
        f"Answer with only one from the list."
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

    for loc in location_list:
        if loc in answer:
            return loc
    return "unknown"

def extract_direct_object_llm_cot(command: str, location: str):
    candidate_objects = [obj for obj, locs in object2locations.items() if location in locs]

    prompt = (
        f"You are a reasoning assistant for grounding object references.\n"
        f"Step 1: Read the command carefully.\n"
        f"Step 2: Identify the object being referred to in the sentence.\n"
        f"Step 3: If it is a synonym or variation, convert it to the most likely canonical object name.\n"
        f"Step 4: Match your final object to one from this list: {list(object2locations.keys())}.\n"
        f"Command: '{command}'\n"
        f"Location: '{location}'\n"
        f"Answer with the final object name only."
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=15)
    obj_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()

    for obj in candidate_objects:
        if obj in obj_prediction:
            return obj

    for obj in object2locations:
        if obj in obj_prediction:
            return obj

    return None

def infer_location(command: str, top_k: int = 3):
    start_time = time.time()
    prompt = (
        f"Choose the best room for this task: {command}\n"
        f"Options: kitchen, bathroom, bedroom, living room, study room.\n"
        f"Answer with the room name only."
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=16)
    raw_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()
    normalized_location = normalize_room_name_llm(raw_prediction)

    top_candidates = []
    with torch.no_grad():
        pred_emb = model.encoder(**tokenizer(raw_prediction, return_tensors="pt")).last_hidden_state.mean(dim=1)
        for loc in location_list:
            loc_emb = model.encoder(**tokenizer(loc, return_tensors="pt")).last_hidden_state.mean(dim=1)
            sim = F.cosine_similarity(pred_emb, loc_emb).item()
            top_candidates.append((loc, sim))
    top_candidates.sort(key=lambda x: x[1], reverse=True)

    elapsed = time.time() - start_time
    return normalized_location, top_candidates[:top_k], elapsed

def get_most_relevant_object(command: str, location: str, top_k: int = 3):
    start_time = time.time()
    candidate_objects = [obj for obj, locs in object2locations.items() if location in locs]
    if not candidate_objects:
        return "no relevant object", [], 0.0

    direct_obj = extract_direct_object_llm_cot(command, location)
    if direct_obj:
        similarities = [(direct_obj, 1.0)] + [(obj, 0.0) for obj in candidate_objects if obj != direct_obj]
        elapsed = time.time() - start_time
        return direct_obj, similarities[:top_k], elapsed

    with torch.no_grad():
        cmd_emb = model.encoder(**tokenizer(command, return_tensors="pt")).last_hidden_state.mean(dim=1)
        similarities = []
        for obj in candidate_objects:
            obj_emb = model.encoder(**tokenizer(obj, return_tensors="pt")).last_hidden_state.mean(dim=1)
            sim = F.cosine_similarity(cmd_emb, obj_emb).item()
            similarities.append((obj, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    best_object = similarities[0][0]
    elapsed = time.time() - start_time
    return best_object, similarities[:top_k], elapsed

def predict_location_and_object(command: str):
    total_start = time.time()
    loc, loc_candidates, loc_time = infer_location(command)
    obj, obj_candidates, obj_time = get_most_relevant_object(command, loc)
    total_elapsed = time.time() - total_start
    return {
        "command": command,
        "location": loc,
        "object": obj,
        "location_candidates": loc_candidates,
        "object_candidates": obj_candidates,
        "location_time": loc_time,
        "object_time": obj_time,
        "total_time": total_elapsed
    }

def process_commands_from_file(file_path):
    locations = []
    objects = []
    total_all = 0.0

    with open(file_path, 'r') as f:
        commands = [line.strip() for line in f if line.strip()]

    for cmd in commands:
        command_start = time.time()
        result = predict_location_and_object(cmd)
        command_time = time.time() - command_start
        total_all += command_time

        locations.append(result['location'])
        objects.append(result['object'])

        print(f"\n🔎 Command: {cmd}")
        print(f"→ Inferred Location: {result['location']} (Time: {result['location_time']:.2f} sec)")
        print("→ Top Location Candidates:")
        for name, score in result["location_candidates"]:
            print(f"   - {name}: {score:.4f}")

        print(f"→ Inferred Object: {result['object']} (Time: {result['object_time']:.2f} sec)")
        print("→ Top Object Candidates:")
        for name, score in result["object_candidates"]:
            print(f"   - {name}: {score:.4f}")

        print(f"⏱️ Total Inference Time for Command: {result['total_time']:.2f} sec")

    print(f"\n⏱️ Total Inference Time for All Commands: {total_all:.2f} sec")
    return locations, objects

file_path = "/home/yunah/LLM/llm_input.txt"
locs, objs = process_commands_from_file(file_path)

print("\nFinal Lists:")
print("Locations:", locs)
print("Objects:  ", objs)
