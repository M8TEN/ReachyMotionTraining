import multiprocessing as mp
import asyncio
from time import sleep

SAMPLE_RATE: float = 1/100.0
PORT = 50056

class Test():
    def __init__(self):
        self.frames = 0
        self.recording = False
        self.terminate: False
    
    def update(self):
        if self.recording:
            self.frames += 1
    
    def start_recording(self):
        print("Starting recording")
        self.frames = 0
        self.recording = True
    
    def stop_recording(self):
        self.recording = False
        if self.frames > 0:
            print(f"Recorded {self.frames} frames")
        else:
            print("No samples recorded")


def update_loop(recorder: Test, run_flag) -> None:
    while run_flag.value != 2:
        if run_flag.value == 1 and not recorder.recording:
            recorder.start_recording()
        elif run_flag.value == 0 and recorder.recording:
            recorder.stop_recording()
        recorder.update()
        sleep(SAMPLE_RATE)

async def establish_connection(ip: str = "127.0.0.1", port=PORT):
    connected: bool = False
    while not connected:
        try:
            reader, writer = await asyncio.open_connection(ip, port)
            connected = True
        except ConnectionRefusedError:
            print("Could not connect, trying again in 5s")
            sleep(5)
    
    return reader, writer

async def wait_on_command(reader: asyncio.StreamReader, recorder_flag):
    while recorder_flag.value != 2:
        command = await reader.read(1)
        command_type = command[0]
        print(f"Command: {command}")
        recorder_flag.value = command_type

async def main(recorder_flag):
    reader, writer = await establish_connection()
    command_task = asyncio.create_task(wait_on_command(reader, recorder_flag))
    print("Connected to Server")
    await command_task

def read_commands(recorder_flag):
    asyncio.run(main(recorder_flag))

if __name__ == "__main__":
    recorder = Test()
    recorder_flag = mp.Value("B", 0)
    recorder_process = mp.Process(target=update_loop, args=(recorder, recorder_flag,))
    command_process = mp.Process(target=read_commands, args=(recorder_flag,))
    recorder_process.start()
    command_process.start()
    recorder_process.join()
    command_process.join()