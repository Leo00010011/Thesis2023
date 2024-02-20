import os
import PySimpleGUI as sg
import threading
from pathlib import Path
try:
    from .data import Patients_Data
    from .gui import Gui
    from .utils import date_str, parse_date, get_patient_folder_name, video_name
    from .bag_utils import compression_state, BagReview
except:
    from tools.bag_utils import compression_state, BagReview
    from tools.data import Patients_Data
    from tools.gui import Gui
    from tools.utils import date_str, parse_date, get_patient_folder_name, video_name


class Pipeline_Item:
    def __init__(self, gui: Gui) -> None:
        self.pipeline: Pipeline = None
        self.index = None
        self.gui = gui

    def process(self, prev_out):
        pass

    def recover(self):
        pass

    def set_pipeline(self, index, pipeline):
        self.index = index
        self.pipeline = pipeline


class B_Select(Pipeline_Item):
    def __init__(self, gui: Gui, options: list[str]) -> None:
        super().__init__(gui)
        self.options = options

    def process(self, prev_out):
        super().process(prev_out)
        sel_option = self.gui.button_chose(self.options)
        return sel_option

    def recover(self):
        super().recover()
        sel_option = self.gui.button_chose(self.options)
        return sel_option


class   Select_Patient(Pipeline_Item):
    def __init__(self, gui: Gui, data: Patients_Data):
        super().__init__(gui)
        self.data = data
        self.selected = None
        self.written_input = None

    def process(self, prev_out):
        super().process(prev_out)
        patients_name = self.data.get_patients()
        event, sel_patient = self.gui.search_chose(patients_name)
        if event != '-SELECT-PATIENT-':
            return event
        self.last_selected = sel_patient
        patient = self.data.get_patient_data(sel_patient)
        return patient

    def recover(self):
        # use de recover data if none raise exception
        super().recover()
        return self.process(None)


class New_Patient(Pipeline_Item):
    def __init__(self, gui: Gui, data: Patients_Data) -> None:
        super().__init__(gui)
        self.data: Patients_Data = data
        self.patient = None

    def process(self, prev_out):
        super().process(prev_out)
        event, data = self.gui.create_patient(
            Patients_Data.check_new_patient_data)
        if event != 'Salvar':
            return event
        self.patient = self.data.add_patient(data)
        self.data.save()
        return self.patient

    def recover(self):
        super().recover()
        return self.process(self.patient)


class Select_Video_by_Date(Pipeline_Item):
    def __init__(self, gui: Gui, data: Patients_Data) -> None:
        super().__init__(gui)
        self.data = data
        self.path = None
        self.sel_key = None
        self.last_patient = None

    def process(self, prev_out):
        super().process(prev_out)
        patient = prev_out
        self.last_patient = patient
        videos = {date_str(video['date']): index for index,
                  video in enumerate(patient['Videos'])}
        event, sel_key = self.gui.search_chose(videos.keys())
        if event != '-SELECT-PATIENT-':
            return event
        self.sel_key = sel_key
        return patient['Videos'][videos[sel_key]]

    def recover(self):
        super().recover()
        return self.process(self.last_patient)


class Playback(Pipeline_Item):
    def __init__(self, gui):
        super().__init__(gui)

    def process(self, prev_out):
        super().process(prev_out)
        video_path = prev_out['path']
        b = BagReview(video_path)
        path = Path(video_path)
        decompress_path = str(path.resolve())[:-4] + '_raw' + '.bag'
        if b.is_compressed():
            text = 'Descomprimiendo el video (Presione \'Atrás\' para cancelar)'
            result = change_compression_asinc(
                'decompress', self.gui, video_path, decompress_path, text)
            if result != 'Ok':
                return result

        if b.is_compressed():
            self.gui.playback(decompress_path)
        else:
            self.gui.playback(video_path)

        if b.is_compressed():
            os.remove(decompress_path)
        return 'Prev'

    def recover(self):
        super().recover()
        return 'Prev'


def change_compression_asinc(comp_func: str, gui: Gui, bag_path: str, new_path: str, text):
    b = BagReview(bag_path)
    if comp_func == 'compress':
        th = threading.Thread(target=b.compress_bag, args=(new_path,))
    else:
        th = threading.Thread(target=b.decompress_bag, args=(new_path,))
    compression_state.reset()
    th.start()
    result = gui.compress(compression_state, text)
    if result != 'Ok':
        compression_state.stop.set()
        th.join()
        return result
    else:
        th.join()
        return 'Ok'


class Record(Pipeline_Item):
    def __init__(self, gui, data: Patients_Data):
        super().__init__(gui)
        self.data = data

    def process(self, prev_out):
        super().process(prev_out)
        # get the right date
        finish_date = False
        while True:
            event, date = self.gui.get_date_for_video()
            if event != 'Comenzar Grabación':
                return event
            try:
                date = parse_date(date)
            except Exception:
                sg.popup_error('''La fecha tiene un formato incorrecto. 
Formato correcto: <Año>-<Mes>-<Dia> <Hora>:<Minuto>:<Segundos>
Ejemplo: 2023-09-09 10:46:45''')
                continue
            break

        # get the path
        data_path = self.data.data_path
        folder_name = get_patient_folder_name(prev_out)
        bag_name = video_name(date) + '_raw' + '.bag'
        final_path = os.path.join(data_path, folder_name, bag_name)
        # record the video
        end_record = False
        event = 'Volver a Grabar'
        while not end_record:
            if event == 'Volver a Grabar':
                event = self.gui.record(final_path)
                if not event or event == 'Prev' or event == 'Exit':
                    return event
            elif event == 'Revisar':
                self.gui.playback(final_path)

            event = self.gui.button_chose(
                ['Volver a Grabar', 'Revisar', 'Guardar Grabación'])
            if event == 'Volver a Grabar':
                os.remove(final_path)
                continue
            elif not event or event == 'Prev' or event == 'Exit':
                os.remove(final_path)
                return event
            elif event == 'Guardar Grabación':
                end_record = True
        compress_path = os.path.join(
            data_path, folder_name, video_name(date) + '.bag')
        text = 'Comprimiendo el video(Presiona atrás para cancelar)'
        result = change_compression_asinc(
            'compress', self.gui, final_path, compress_path, text)
        if result == 'Ok':
            if os.path.exists(final_path):
                os.remove(final_path)
            final_path = compress_path
        prev_out['Videos'].append({'path': final_path, 'date': date})
        compression_state.reset()
        self.data.save()
        if result == 'Ok':
            return 'Prev'
        else:
            return result

    def recover(self):
        super().recover()
        return 'Prev'


class Compress(Pipeline_Item):
    def __init__(self, gui, data: Patients_Data):
        super().__init__(gui)
        self.data = data

    def process(self, prev_out):
        super().process(prev_out)
        # Buscar los videos que están sin comprimir
        uncompressed_bags = []
        for patient in self.data.get_patients():
            for video in self.data.get_patient_video_info(patient):
                video_path = video['path']
                if os.path.exists(video_path) and not BagReview(video_path).is_compressed():
                    uncompressed_bags.append(video)
        # Comprimir los videos
        help_str = 'Comprimiendo los videos, presione \'Atrás\' para detener la aplicación\n'
        lines = ['->' + video['path'] + '\n' for video in uncompressed_bags]
        for video in uncompressed_bags:
            path = video['path']
            comp_path = os.path.join(os.path.split(
                path)[0], video_name(video['date']) + '.bag')
            text = ''.join([help_str, *lines])
            result = change_compression_asinc(
                'compress', self.gui, path, comp_path, text)
            if result != 'Ok':
                return result
            lines.pop(0)
            video['path'] = comp_path
            self.data.save()
            if os.path.exists(path):
                os.remove(path)
        return 'Prev'

    def recover(self):
        super().recover()
        return 'Prev'


class Bifurcation(Pipeline_Item):
    def __init__(self, tale_dict: dict[(str, list[Pipeline_Item])], inside_item: Pipeline_Item) -> None:
        super().__init__(inside_item.gui)
        self.tale_dict = tale_dict
        self.inner_item = inside_item

    def process(self, prev_out):
        super().process(prev_out)
        if not self.pipeline:
            raise Exception('A Pipeline_Item must be inside a Pipeline')
        result = self.inner_item.process(prev_out)

        try:
            tale = self.tale_dict[result]
        except KeyError:
            return result

        self.pipeline.item_list = self.pipeline.item_list[:self.index + 1] + tale
        return result

    def recover(self):
        super().recover()
        if not self.pipeline:
            raise Exception('A Pipeline_Item must be inside a Pipeline')
        result = self.inner_item.recover()

        try:
            tale = self.tale_dict[result]
        except KeyError:
            return result

        self.pipeline.item_list = self.pipeline.item_list[:self.index + 1] + tale
        return result

    def set_pipeline(self, index, pipeline):
        super().set_pipeline(index, pipeline)
        self.inner_item.set_pipeline(index, pipeline)
        for tale in self.tale_dict.values():
            for inside_index, pipe_item in enumerate(tale):
                pipe_item.set_pipeline(index + 1 + inside_index, pipeline)


class Pipeline:
    def __init__(self, gui: Gui, items_list: list[Pipeline_Item]) -> None:
        self.gui = gui
        self.item_list = items_list
        for index, item in enumerate(items_list):
            item.set_pipeline(index, self)

    def start(self):
        self.gui.start()
        index = -1
        result = 'nothing'
        while not (not result or result == 'Exit'):
            if result == 'Prev':
                index = max(0, index - 1)
                if index != 0:
                    self.gui.set_back_visibility(True)
                result = self.item_list[index].recover()
                self.gui.set_back_visibility(False)
            else:
                index = (index + 1) % len(self.item_list)
                if index != 0:
                    self.gui.set_back_visibility(True)
                result = self.item_list[index].process(result)
                self.gui.set_back_visibility(False)
        self.gui.end()
