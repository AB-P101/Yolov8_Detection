from datetime import datetime
now = datetime.today()
Floder_Log = now.strftime("_%d_%m_%Y")
now = now.strftime("Date : %d/%m/%Y Time : %H:%M:%S")

def append_to_note(filename, content):
    with open(filename, 'a') as file:
        file.write(content + '\n')

append_to_note("Log/Log"+str(Floder_Log)+".txt", "error >> Camara1 >> NG, " + str(now))

print(now)