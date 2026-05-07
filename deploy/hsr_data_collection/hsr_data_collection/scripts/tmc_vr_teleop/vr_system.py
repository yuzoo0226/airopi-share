import time
import json
import socket
import struct

import triad_openvr


def send_message(client_socket, msg):
    header = struct.pack('!I', len(msg))
    client_socket.send(header + msg.encode())

def receive_message(socket):
    # Read the header (2 bytes) to find out the length of the message
    header = socket.recv(2)
    if not header:
        raise ValueError("Connection closed by the server")
    message_length = int.from_bytes(header, byteorder='big')
    # Read the message based on its length
    message = socket.recv(message_length)
    return message.decode()

def connect_to_server(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            client_socket.connect((host, port))
            print("Server connected.")
            return client_socket
        except ConnectionRefusedError:
            print("Server is not yet available. Retrying in 5 seconds...")
            time.sleep(5)

def initialize_vr_system():
    """Initialize VR system and print discovered objects."""
    vr_system = triad_openvr.triad_openvr()
    vr_system.print_discovered_objects()
    return vr_system

def get_device_pose_and_inputs(device):
    """Extract pose and inputs from a VR device."""
    pose = [[x for x in row] for row in device.get_pose_matrix()]
    inputs = device.get_controller_inputs()
    return pose, inputs

def main(host='localhost', port=8000, frequency=50):
    vr_system = initialize_vr_system()
    client_socket = connect_to_server(host, port)

    try:
        interval = 1 / frequency
        step = 0
        while True:
            start = time.time()

            data = {}
            for device_id in ['controller_1', 'controller_2', 'hmd_1']:
                if device_id in vr_system.devices:
                    device = vr_system.devices[device_id]
                    pose, inputs = get_device_pose_and_inputs(device)
                    data[device_id] = {'pose': pose}
                    if 'controller' in device_id:
                        data[device_id]['inputs'] = inputs

            send_message(client_socket, json.dumps(data))

            wrist_force = float(receive_message(client_socket))
            if wrist_force > 15.0:
                rate = int(70 / wrist_force)
                if rate == 0 or step % rate == 0:
                    vr_system.devices["controller_1"].trigger_haptic_pulse(duration_micros=1000)

            step += 1
            time.sleep(max(0, interval - (time.time() - start)))

    except (ConnectionResetError, ValueError) as e:
        print("Connection lost. Exiting.")
    finally:
        client_socket.close()

if __name__ == "__main__":
    main()