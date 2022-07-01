import json
  
# Opening JSON file
# f = open('result_face.json')

# returns JSON object as a dictionary
# data = json.load(f)

face_input_file_name = 'result_face_3.json'
action_input_file_name = 'result_action.json'
output_file_name = "result_integrated_3.json"

# READ face.json
data = []
count = 0
with open(face_input_file_name, 'r') as f:
    for line in f:
        data.append(json.loads(line))
        count += 1

number_of_frame = count // 2                       # 3

integrated_line_list = []
#print(type(integrated_line_list))

# iteration: 0 ~ 2
for i in range(number_of_frame):                   # 0, 1, 2

  #print(2 * i + 1)                                  # 1, 3, 5

  iindex = data[2 * i]["index"]["_index"]           # 0, 2, 4: _index,  e.g. vid_*
  frame_number = data[2 * i + 1]["frame_number"]                    # 1, 3, 5: frame_number,      e.g. 1,2,3
  face_description = data[2 * i + 1]["label"][0]["description"]     # 1, 3, 5: face_description,  e.g. Tom
  distance = data[2 * i + 1]["label"][0]["score"]                # 1, 3, 5: distance,          e.g. 0.1

  ## =================== ##   add!!!!!!!!!!!!!!!!!
  action_description = "___hacking___"
  confidence_score = 0.01
  ## =================== ## 

  integrated_dict_header_inner = {
      "_index": iindex,
      "_type": "frame",
      "_id": i + 1                              # 1, 2, 3
  }
  integrated_dict_header = {
    "index": integrated_dict_header_inner,
  }
  # INSERT header line
  integrated_line_list.append(integrated_dict_header)
  
  integrated_dict_body_label = [{
      "face_description": face_description,
      "action_description": action_description,
      "confidence_score": confidence_score,
      "distance": distance
  }]
  integrated_dict_body = {
    "frame_number": frame_number,
    "label": integrated_dict_body_label
  }
  # INSERT header line
  integrated_line_list.append(integrated_dict_body)

  #print(integrated_dict_header)
  #print(integrated_dict_body)

# Closing file
f.close()

"""
print(data[1]["label"][0]["description"])       # face_description
print(data[1]["label"][0])                      # label contents
print(data[1]["label"])                         # {label contents}
print(data[1])                                  # body (odd number)

print("===")
print(data[0]["index"]["_index"])               # _index: e.g. vid_0
print("===")
print(data[1]["frame_number"])                  # frame_number
print(data[1]["label"][0]["description"])       # face_description
print(data[1]["label"][0]["distance"])          # distance

iindex = data[0]["index"]["_index"]
"""

"""
integrated_dict_header_inner = {
    "_index": "vid_",
    "_type": "frame",
    "_id": 0
}
integrated_dict_header = {
  "index": integrated_dict_header_inner,
}
print(integrated_dict_header)

integrated_dict_body_label = [{
    "face_description": "John_Doe",
    "action_description": "hacking",
    "confidence_score": 0.01,
    "distance": 0.01
}]
integrated_dict_body = {
  "frame_number": 0,
  "label": integrated_dict_body_label
}
print(integrated_dict_body)
"""

# TODO: READ action.json
import os

f = open('action_raw.txt')

lines = f.readlines()

dict_list = list()
action_dict = dict()

for i in range(29):
    action_dict = dict()
    action_dict['frame_number'] = i
    action_dict['action_description'] = lines[3].strip('\n')
    #print(lines[3].strip('\n'))
    action_dict['confidence_score'] = lines[4].strip('\n')
    #print(lines[4].strip('\n'))

    dict_list.append(action_dict)

for i in range(len(lines)):

    # Make dictionary
    if i % 5 == 0:
        action_dict = dict()

    # frame_number
    if (i + 1) % 5 == 2:
        #print(i)
        #print(lines[i - 1])
        action_dict['frame_number'] = int(lines[i].strip('\n')) - 1
    
    # action_description
    if (i + 1) % 5 == 4:
        #print(i)
        #print(lines[i -1])
        action_dict['action_description'] = lines[i].strip('\n')

    # confidence_score
    if (i + 1) % 5 == 0:
        #print(i)
        #print(lines[i -1])
        action_dict['confidence_score'] = lines[i].strip('\n')

        dict_list.append(action_dict)


    # if i == 20:
    #     break

    #print(action_dict)

    # for j in range(3):
    #     pass
    # for line in lines:
    #     count += 1
    #     print(count)

# for i in range((len(lines) + 145) // 5):
#     print(dict_list[i])

#print(number_of_frame)

for i in range(number_of_frame):                   # 0, 1, 2
  #print(dict_list[i]['frame_number'])
  ## =================== ##   add!!!!!!!!!!!!!!!!!
  action_description = dict_list[i]['action_description']
  confidence_score = dict_list[i]['confidence_score']
  #print(action_description)
  #print(confidence_score)
  ## =================== ## 
  #print(i * 2 + 1)
  #print(integrated_line_list[i * 2 + 1]["label"][0]["face_description"])

  integrated_line_list[i * 2 + 1]["label"][0]["action_description"] = action_description      # action_description
  integrated_line_list[i * 2 + 1]["label"][0]["confidence_score"] = confidence_score          # confidence_score


# WRITE phase
print("Write \'" + output_file_name + "\'!")

# for i in range(count):
#   print(integrated_line_list[i])

# WRITE result_integrated_sample.json
with open(output_file_name, 'a') as f:
  for dict in integrated_line_list:
      string = json.dumps(dict)
      f.write(string)
      f.write("\n")

print("Write \'" + output_file_name + "\' complete!")  

# Closing file
f.close()