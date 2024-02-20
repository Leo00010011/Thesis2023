import os
import json
import glob
try:
    from .utils import parse_date,fecha_dict_to_str,get_patient_folder_name
except:
    from tools.utils import parse_date, fecha_dict_to_str,get_patient_folder_name

class MoreDataExpected(Exception):
    def __init__(self, count, expected) -> None:
        self.message = f"Se esperaban {expected} datos pero se dieron {count}"
        super().__init__(self.message)


class WrongDateFormat(Exception):
    def __init__(self) -> None:
        self.message = '''La fecha tiene un formato incorrecto. 
Formato correcto: <Año>-<Mes>-<Dia> <Hora>:<Minuto>:<Segundos>
Ejemplo: 2023-09-09 10:46:45'''
        super().__init__(self.message)


class NameMustBeAlpha(Exception):
    def __init__(self) -> None:
        self.message = "El nombre solo debe tener letras y espacios"
        super().__init__(self.message)


class AgeMustBeNumber(Exception):
    def __init__(self) -> None:
        self.message = "La edad solo debe tener números entre 1 y 130"
        super().__init__(self.message)


class InvalidOption(Exception):
    def __init__(self, option_name, valid_options) -> None:
        v_options = ','.join(valid_options)
        self.message = f"Los valores validos para {option_name} son [{v_options}]"
        super().__init__(self.message)


class InvalidSex(InvalidOption):
    def __init__(self) -> None:
        super().__init__("sexo", ["Hembra", "Varón"])


class InvalidSkinColor(InvalidOption):
    def __init__(self) -> None:
        super().__init__("color de piel", ["Blanco", "Negro"])


class InvalidDiebetesType(InvalidOption):
    def __init__(self) -> None:
        super().__init__("tipo de diabetes", [
            "Tipo I", "Tipo II", "No diabetes"])


class InvalidUlcerType(InvalidOption):
    def __init__(self) -> None:
        super().__init__("tipo de ulcera", [
            "Neuroinfecciosa", "Isquémica", "Flebolinfática", "Combinada"])


class InvalidLocalization(InvalidOption):
    def __init__(self) -> None:
        super().__init__("localización", ["Pie Derecho", "Pie Izquierdo"])


class Patients_Data:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def load(self):
        # Si no existe data crear uno nuevo en ese path
        pass

    def check_new_patient_data(data):
        try:
            Name, Age, Sex, Skin_Color, Diabetes_Type, Ulcer_Type, Localization, Date = data
        except Exception:
            return MoreDataExpected(len(data), 8)

        if not all([word.isalpha() or len(word) == 0 for word in Name.split(' ')]):
            return NameMustBeAlpha

        try:
            if int(Age) < 1 or int(Age) > 130:
                return AgeMustBeNumber
        except ValueError:
            return AgeMustBeNumber

        if not Sex in {'Hembra', 'Varon'}:
            return InvalidSex

        if not Skin_Color in {'Blanco', 'Negro'}:
            return InvalidSkinColor

        if not Diabetes_Type in {"Tipo I", "Tipo II", "No diabetes"}:
            return InvalidDiebetesType

        if not Ulcer_Type in {"Neuroinfecciosa", "Isquémica", "Flebolinfática", "Combinada"}:
            return InvalidUlcerType

        if not Localization in {"Pie Derecho", "Pie Izquierdo"}:
            return InvalidLocalization

        try:
            parse_date(Date)
        except Exception:
            return WrongDateFormat

    def get_patient_data(self, patient_id):
        pass

    def add_patient(self, data):
        pass

    def get_patient_video_info(self, patient):
        return 

    def get_patients(self) -> list[str]:
        pass

    def save(self):
        pass


class Patients_Data_Simple(Patients_Data):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)
        self.data = None
        self.patient_index_map = None

    def add_patient_to_index_map(self, name, index):
        if self.patient_index_map.get(name, None) == None:
            self.patient_index_map[name] = index
        else:
            self.patient_index_map[f'{name}-{index}'] = index

    def load(self):
        super().load()
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
            self.patient_index_map = dict()
            for i in range(len(self.data)):
                current_name = self.data[i]['Name']
                self.add_patient_to_index_map(current_name, i)
        else:
            self.data = []
            self.patient_index_map = dict()

    def add_patient(self, data):
        super().add_patient(data)
        Name, Age, Sex, Skin_Color, Diabetes_Type, Ulcer_Type, Localization, date = data
        parsed_date = parse_date(date)
        patient_data = {
            'Name': Name,
            'Age': Age,
            'Sex': Sex,
            'Skin_Color': Skin_Color,
            'Diabetes_Type': Diabetes_Type,
            'Ulcer_Type': Ulcer_Type,
            'Localization': Localization,
            'First_Date': parsed_date,
            'Videos': []
        }
        self.data.append(patient_data)
        self.add_patient_to_index_map(Name, len(self.data) - 1)
        return patient_data

    def get_patient_data(self, patient_id):
        super().get_patient_data(patient_id)
        index = self.patient_index_map[patient_id]
        return self.data[index]

    def get_patient_video_info(self, patient_id):
        super().get_patient_video_info(patient_id)
        index = self.patient_index_map[patient_id]
        return self.data[index]['Videos']

    def get_patients(self) -> list[str]:
        super().get_patients()
        return self.patient_index_map.keys()

    def save(self):
        super().save()
        data_s = json.dumps(self.data)
        with open(self.data_path, "w") as f:
            f.write(data_s)


class Patients_Data_Splited(Patients_Data_Simple):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)

    def load(self):
        self.patient_index_map = dict()
        self.data = []
        for index, path in enumerate(glob.glob(f'{self.data_path}/*/')):
            with open(os.path.join(path, 'data.json'), 'r') as f:
                patient_data = json.load(f)
                self.data.append(patient_data)
                self.add_patient_to_index_map(patient_data['Name'], index)

    def save(self):
        for patient in self.data:
            folder_name = get_patient_folder_name(patient)
            if not os.path.exists(os.path.join(self.data_path, folder_name)):
                os.mkdir(os.path.join(self.data_path, folder_name))
            with open(os.path.join(self.data_path, folder_name, 'data.json'), 'w') as f:
                json.dump(patient, f)



def migrate_from_simple_to_splitted(ori_path, dest_path):
    data = None
    with open(ori_path, 'r') as f:
        data = json.load(f)
    for patient in data:
        nombre = patient['Name']
        fecha = fecha_dict_to_str(patient['First_Date'])
        folder_name = f'{fecha}-{nombre}'
        folder_path = os.path.join(dest_path, folder_name)
        os.mkdir(folder_path)
        with open(os.path.join(folder_path, 'data.json'), 'w') as f:
            json.dump(patient, f)
