with open('washington_content.txt','r')as f,open('content.txt','w')as f2:
    lines=f.readlines()
    for index,line in enumerate(lines):
        f2.write(str(index)+line)