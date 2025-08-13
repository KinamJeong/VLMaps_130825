from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn.functional as F
import time
import re
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
    "cabinet": ["study room", "kitchen", "bedroom"], "mirror":["bathroom"], "book":["study room"]
}

mp3dcat = [
    "void", "table", "sofa", "bed", "sink", "chair", "toilet", "tv_monitor", "wall", "window",
    "clothes", "picture", "cushion", "coffeepot", "appliances", "tree", "shelving", "door",
    "cabinet", "curtain", "chest_of_drawers", "carpet", "bookcase", "desktop_computer",
    "dishrag", "carpet", "study_desk", "counter", "oscilloscope", "lighting", "stove",
    "book", "blinds", "refrigerator", "seating", "microwave", "furniture", "towel",
    "mirror", "shower"
]


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
        f"Command: '{command}'\n"
        f"Location: '{location}'\n"
        f"Which object is most relevant to the command above?"
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

def get_object_from_mp3dcat(command: str):
    command = command.lower()
    for obj in mp3dcat:
        pattern = r"\b" + re.escape(obj.lower()) + r"\b"
        if re.search(r"\b" + re.escape(obj.lower()) + r"\b", command):
            print(f"✓ Directly matched object from MP3D: '{obj}' in command: '{command}'")
            print("→ MP3D object found in command. LLM object inference skipped.")
            return obj
    print(f"✗ No MP3D object matched directly in command: '{command}'. Proceeding with LLM inference.")
    return None

def infer_location(command: str, top_k: int = 3):
    start_time = time.time()
    prompt = (
        f"Which room is best suited for this command?\n"
        f"Command: {command}\n"
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

    direct_obj = get_object_from_mp3dcat(command)
    loc, loc_candidates, loc_time = infer_location(command)

    if direct_obj:
        total_elapsed = time.time() - total_start
        return {
            "command": command,
            "location": loc,
            "object": direct_obj,
            "location_candidates": loc_candidates,
            "object_candidates": [(direct_obj, 1.0)],
            "location_time": loc_time,
            "object_time": 0.0,
            "total_time": total_elapsed
        }

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
    start_all = time.time()

    with open(file_path, 'r') as f:
        commands = [line.strip() for line in f if line.strip()]

    for cmd in commands:
        result = predict_location_and_object(cmd)

        locations.append(result['location'])
        objects.append(result['object'])

        #print(f"\n🔎 Command: {cmd}")
        #print(f"→ Inferred Location: {result['location']} (Time: {result['location_time']:.2f} sec)")
        #print("→ Top Location Candidates:")
        #for name, score in result["location_candidates"]:
        #    print(f"   - {name}: {score:.4f}")

        #print(f"→ Inferred Object: {result['object']} (Time: {result['object_time']:.2f} sec)")
        #print("→ Top Object Candidates:")
        #for name, score in result["object_candidates"]:
            #print(f"   - {name}: {score:.4f}")

        #print(f"⏱️ Total Inference Time for Command: {result['total_time']:.2f} sec")

    total_all = time.time() - start_all
    #print(f"\n⏱️ Total Inference Time for All Commands: {total_all:.2f} sec")
    return locations, objects

# 🔍 Run on uploaded file
#file_path = "/home/ivl1/chatbot_server/chatlogs_eng/llm_sentence.txt" # from uploaded file
#locs, objs = process_commands_from_file(file_path)

#print("\nFinal Lists:")
#print("Locations:", locs)
#print("Objects:  ", objs)
