import asyncio
from RecordMovement import MotionRecorder

PORT: int = 50056
SAMPLE_RATE: float = 1/100.0

RECORDING_STOP: int = 0
RECORDING_START: int = 1
RECORDING_END: int = 2
NO_REQUEST: int = 3
ALLOW_REQUESTS: int = 4
CLOSE_CONNECTION: int = 5

client_reader = None
client_writer = None
connection_event = asyncio.Event()

async def establish_connection(port=PORT):
    print("Starting Server")
    await asyncio.start_server(on_client_connected, port=port)

def on_client_connected(reader, writer):
    print("Client connected")
    global client_reader, client_writer, connection_event
    client_reader = reader
    client_writer = writer
    connection_event.set()

async def wait_on_command(reader: asyncio.StreamReader, start_event: asyncio.Event, stop_event: asyncio.Event, halt_event: asyncio.Event):
    while not halt_event.is_set():
        command = await reader.read(1)
        command_type = command[0]
        print(f"Command: {command}")
        if not start_event.is_set() and command_type == RECORDING_START:
            start_event.set()
        elif command_type == RECORDING_STOP:
            stop_event.set()
        elif command_type == CLOSE_CONNECTION:
            print("Connection closed by Client")
            halt_event.set()
            start_event.set()
            stop_event.set()

async def main():
    start_event = asyncio.Event()
    stop_event = asyncio.Event()
    halt_event = asyncio.Event()
    recorder = MotionRecorder(sample_rate=SAMPLE_RATE)
    await establish_connection()
    await connection_event.wait()
    await recorder.send_client_command(client_writer, bytes([ALLOW_REQUESTS]))
    update_task = asyncio.create_task(recorder.update(halt_event))
    start_task = asyncio.create_task(recorder.start_recording(client_writer, start_event, halt_event))
    stop_task = asyncio.create_task(recorder.stop_recording(client_writer, stop_event, halt_event))
    command_task = asyncio.create_task(wait_on_command(client_reader, start_event, stop_event, halt_event))
    await start_task
    await stop_task
    await update_task
    await command_task

if __name__ == "__main__":
    asyncio.run(main())