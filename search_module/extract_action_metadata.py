import os

f = open('action_raw.txt')

lines = f.readlines()

dict_list = list()
action_dict = dict()

for i in range(29):
    action_dict = dict()
    action_dict['frame_number'] = i
    action_dict['action_description'] = lines[3].strip('\n')
    print(lines[3].strip('\n'))
    action_dict['confidence_score'] = lines[4].strip('\n')
    print(lines[4].strip('\n'))

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

for i in range((len(lines) + 145) // 5):
    print(dict_list[i])