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


def infer_location(command: str, top_k: int = 3):
    start_time = time.time()

    prompt = (
        f"Choose the best room for this task: {command}\n"
        f"Options: kitchen, bathroom, bedroom, living room, study room.\n"
        f"Answer with the room name only."
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=16)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()

    # 정규화된 포함 조건으로 변경
    prediction_clean = prediction.replace('.', '').replace(',', '').strip()
    matches = [loc for loc in location_list if loc in prediction_clean]

    best_location = matches[0] if matches else "unknown"

    # 유사도 ranking은 그대로
    top_candidates = []
    with torch.no_grad():
        pred_emb = model.encoder(**tokenizer(prediction, return_tensors="pt")).last_hidden_state.mean(dim=1)
        for loc in location_list:
            loc_emb = model.encoder(**tokenizer(loc, return_tensors="pt")).last_hidden_state.mean(dim=1)
            sim = F.cosine_similarity(pred_emb, loc_emb).item()
            top_candidates.append((loc, sim))
    top_candidates.sort(key=lambda x: x[1], reverse=True)

    elapsed = time.time() - start_time
    return best_location, top_candidates[:top_k], elapsed



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
    top_candidates = similarities[:top_k]
    best_object = top_candidates[0][0]
    elapsed = time.time() - start_time
    return best_object, top_candidates, elapsed


def predict_location_and_object(command: str):
    loc, loc_candidates, loc_time = infer_location(command)
    obj, obj_candidates, obj_time = get_most_relevant_object(command, loc)
    return {
        "command": command,
        "location": loc,
        "object": obj,
        "location_candidates": loc_candidates,
        "object_candidates": obj_candidates
    }


# 🔹 main: read from text file
def process_commands_from_file(file_path):
    locations = []
    objects = []

    with open(file_path, 'r') as f:
        commands = [line.strip() for line in f if line.strip()]

    for cmd in commands:
        result = predict_location_and_object(cmd)
        locations.append(result['location'])
        objects.append(result['object'])

        print(f"\nCommand: {result['command']}")
        print(f"→ Inferred Location: {result['location']}")
        print("→ Top Location Candidates:")
        for name, score in result["location_candidates"]:
            print(f"   - {name}: {score:.4f}")

        print(f"→ Inferred Object: {result['object']}")
        print("→ Top Object Candidates:")
        for name, score in result["object_candidates"]:
            print(f"   - {name}: {score:.4f}")

    return locations, objects


# 🔍 Run on uploaded file
file_path = "/home/ivl1/chatbot_server/chatlogs_eng/llm_sentence.txt" # from uploaded file
locs, objs = process_commands_from_file(file_path)

print("\nFinal Lists:")
print("Locations:", locs)
print("Objects:  ", objs)
