import serial as ser

from Action import Action




class ESP32DeviceManager:
    LIGHTS_MUSIC = "L4\n"
    LIGHTS_VOICE = "L3\n"
    LIGHTS_ON = "L2\n"
    LIGHTS_IDLE = "L0\n"
    LIGHTS_OFF = "L1\n"


    LEDInstructions = [(LIGHTS_ON,
                        "Turn on the lights to a certain brightness(use arg1 to set brightness), use it when the user says 'study lights' or 'turn the lights on'",
                        "Set this to the brightness level the user wants (0-255), if not specified set it to 128",
                        "Does nothing"), 
                        (LIGHTS_OFF,
                        "Turn off the lights, use it when the user says 'off the lights' or 'turn the lights off'",
                        "Does nothing",
                        "Does nothing"), 
                        (LIGHTS_IDLE,
                         "Plays the rgb idle animation, use it when the user says 'set the LEDs to idle mode', or 'RGB lights'",
                         "Does nothing",
                         "Does nothing")]


   
    def __init__(self, device:str):

        self.LEDstate = ESP32DeviceManager.LIGHTS_OFF


        self.device = device
        
        self.serial = ser.Serial(
                    port=device,
                    baudrate=115200,       
                    parity=ser.PARITY_NONE,
                    stopbits=ser.STOPBITS_ONE,
                    bytesize=ser.EIGHTBITS,
                    timeout=1           
                )

        self.connect()



    def get_connection_status(self): return self.connected

    def connect(self):

        if (not self.serial.is_open): self.serial.open()

        if (self.serial.is_open == False):
            raise RuntimeError("Failed to open serial port")
            return

        self.serial.write(b"@")

        if (self.serial.readline().strip()): 
            self.connected = True
            return
        
        raise RuntimeError("Device not responding")

    def disconnect(self):
        self.serial.close()  
        self.connected = False

    def send_command(self, command:str):
        if not self.connected:
            raise Exception("Device not connected")
        self.serial.write(command.encode())

    def receive_response(self):
        if not self.connected:
            raise Exception("Device not connected")
        response = self.serial.readline().decode().strip()
        return response


    def setLEDState(self, state:str, arg1:float = 0, arg2:float = 0, check=True):
        if (self.serial.is_open == False): return False
        self.LEDstate = state
        self.send_command(f"{self.LEDstate}{arg1}\n{arg2}\n")
        if (check):
            is_ok = True
            while (is_ok):
                response = self.receive_response()
                if (response == "k"): is_ok = False
                elif (response == "n"): return False
        return True


    def setLEDState(self, state:int, arg1:float, arg2:float = 0): 
        if (self.serial.is_open == False): return False
        self.LEDstate = ESP32DeviceManager.LEDInstructions[state][0]
        self.send_command(f"{self.LEDstate}{arg1}\n{arg2}\n")

        is_ok = True
        while (is_ok):
            response = self.receive_response()
            if (response == "k"): is_ok = False
            elif (response == "n"): return False


        return True



    def getLEDState(self): return self.LEDstate    



    def actions(self):
        led_state_description = ""
        led_arg1_description = ""
        led_arg2_description = ""

        for i, item in enumerate(ESP32DeviceManager.LEDInstructions):
            led_state_description += f"{i} : {item[1]}, "
            led_arg1_description += f"{i} : {item[2]}, "
            led_arg2_description += f"{i} : {item[3]}, "

        print(led_arg1_description)
        print(led_arg2_description)


        return [
            Action("Set the state of the leds, call this when the user wants to turn on/off the lights or set it to a state, the output is a boolean indicating if the command was successful or not", 
                   self.setLEDState,
                   describe={
                        "state": f"Set this to one of the following numbers, read the instructions for each: {led_state_description}",
                        "arg1": f"Set this to one of the following depending on the state (so if you decide the state is 0, then follow the intructions of 0 to figure out what to set it): {led_arg1_description}",
                        "arg2": f"Set this to one of the following depending on the state (so if you decide the state is 0, then follow the intructions of 0 to figure out what to set it): {led_arg2_description}"
                    })
        ]





if __name__ == "__main__":
    device_manager = ESP32DeviceManager("/dev/ttyUSB0")
    success = device_manager.setLEDState(0, arg1=125)

    device_manager.actions()

    # print(f"Set LED state success: {success}")

    device_manager.disconnect()
    