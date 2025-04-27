import pyspacemouse
import time

def print_spacemouse_state():
    # Try to open the SpaceMouse device
    success = pyspacemouse.open(dof_callback=pyspacemouse.print_state, 
                              button_callback=pyspacemouse.print_buttons)
    
    if success:
        print("SpaceMouse connected successfully!")
        # Continuously read and print the state
        while True:
            state = pyspacemouse.read()
            print(state.x, state.y, state.z, state.roll, state.pitch, state.yaw)
            time.sleep(0.01)  # Small delay to prevent overwhelming the system
    else:
        print("Failed to connect to SpaceMouse device")

if __name__ == "__main__":
    print_spacemouse_state()
