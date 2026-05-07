import re
import random

class CommandGenerator:
    def __init__(self, location_names, object_names):
        self.location_names = location_names
        self.object_names = object_names
        self.verb_dict = {
            "take": ["take", "get", "grasp", "pick up"],
            "place": ["put", "place"],
        }

    def generate_command(self, has_object=False, object=""):
        command_string = ""
        object_name = ""
        if not has_object:
            command_string = "{takeVerb} {art} {obj} {takeLoc}"
        else:
            command_string = "{placeVerb} {art} "+ object + " {plcmtLoc}"
        for ph in re.findall(r'(\{\w+\})', command_string, re.DOTALL):
            replacement_word = self.insert_placeholders(ph)
            if ph=="{obj}":
                object_name = replacement_word
            command_string = command_string.replace(ph, replacement_word)
        art_ph = re.findall(r'\{(art)\}\s*([A-Za-z])', command_string, re.DOTALL)
        if art_ph:
            command_string = command_string.replace("art", "an" if art_ph[0][1].lower() in ["a", "e", "i", "o", "u"] else "a")
        return command_string.replace('{','').replace('}','').replace('_', ' '), object_name

    def insert_placeholders(self, ph):
        ph = ph.replace('{', '').replace('}', '')
        if len(ph.split('_')) > 1:
            ph = random.choice(ph.split('_'))
        if ph == "takeVerb":
            return random.choice(self.verb_dict["take"])
        elif ph == "placeVerb":
            return random.choice(self.verb_dict["place"])
        elif ph == "obj":
            return random.choice(self.object_names)
        elif ph == "takeLoc":
            takable_location_names =  [location for location in self.location_names if not location.startswith("in")]
            return random.choice(takable_location_names)
        elif ph == "plcmtLoc":
            return random.choice(self.location_names)
        elif ph == "art":
            return "{art}"
        else:
            return "WARNING"