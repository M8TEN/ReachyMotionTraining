import asyncio
from time import sleep
#from RecordMovement import MotionRecorder

PORT: int = 50056
SAMPLE_RATE: float = 1/100.0

class Test():
    def __init__(self):
        self.frames = 0
        self.recording = False
    
    async def update(self, halt_event: asyncio.Event):
        while not halt_event.is_set():
            if self.recording:
                self.frames += 1
            await asyncio.sleep(SAMPLE_RATE)
    
    async def start_recording(self, start_event: asyncio.Event, halt_event: asyncio.Event):
        while not halt_event.is_set():
            await start_event.wait()
            print("Starting recording")
            self.frames = 0
            self.recording = True
    
    async def stop_recording(self, stop_event: asyncio.Event, halt_event: asyncio.Event):
        while not halt_event.is_set():
            await stop_event.wait()
            self.recording = False
            if self.frames > 0:
                print(f"Recorded {self.frames} frames")
            else:
                print("No samples recorded")

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

async def wait_on_command(reader: asyncio.StreamReader, start_event: asyncio.Event, stop_event: asyncio.Event, halt_event: asyncio.Event):
    while not halt_event.is_set():
        command = await reader.read(1)
        command_type = command[0]
        print(f"Command: {command}")
        if not start_event.is_set() and command_type == 1:
            start_event.set()
            start_event.clear()
        elif command_type == 0:
            stop_event.set()
            stop_event.clear()
        elif command_type == 2:
            print("Connection closed by Server")
            halt_event.set()
            start_event.set()
            stop_event.set()

async def main():
    start_event = asyncio.Event()
    stop_event = asyncio.Event()
    halt_event = asyncio.Event()
    recorder = Test()
    update_task = asyncio.create_task(recorder.update(halt_event))
    start_task = asyncio.create_task(recorder.start_recording(start_event, halt_event))
    stop_task = asyncio.create_task(recorder.stop_recording(stop_event, halt_event))
    reader, writer = await establish_connection()
    command_task = asyncio.create_task(wait_on_command(reader, start_event, stop_event, halt_event))
    print("Connected to Server")
    await start_task
    await stop_task
    await update_task
    await command_task


asyncio.run(main())