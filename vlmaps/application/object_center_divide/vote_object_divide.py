import os
import json
import google.generativeai as genai

genai.configure(api_key="AIzaSyCQZYf4_IycN7k7NlQGWhTKzGFOIW9GK9w")
model = genai.GenerativeModel("gemini-1.5-flash")

valid_rooms = ["Bathroom", "Bedroom", "LivingRoom", "Kitchen", "Study room"]

mp3dcat = [
    "table","sofa","bed","sink","chair","toilet","tv_monitor","wall","window","clothes","picture","cushion","coffeepot","appliances","tree","shelving",
    "door","cabinet","curtain","chest_of_drawers","carpet","bookcase","desktop_computer","dishrag","carpet","study_desk","counter","oscilloscope",
    "lighting","stove","book","blinds","refrigerator","seating","microwave","furniture","towel","mirror","shower"
]

prompt = (
    "다음 객체 리스트에 대해, 각 객체가 포함될 수 있는 방을 가능한 경우마다 1개 이상 최대 3개까지"
    " Bathroom, Bedroom, LivingRoom, Kitchen, Study room 중에서 추천해주세요.\n"
    f"객체: {mp3dcat}\n"
    "JSON 배열 형식으로 반환해주세요: "
    "[{\"object_name\": ..., \"room_votes\": [...]}]"
) 

response = model.generate_content(prompt)
text = response.text

try:
    object_to_room_votes = {
        item["object_name"]: item["room_votes"]
        for item in json.loads(text)
    }
    print(json.dumps(object_to_room_votes, indent=2))
except Exception as e:
    print("[ERROR] JSON 파싱 실패:", e)
    print("[Raw Gemini Output]:\n", text)