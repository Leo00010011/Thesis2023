import PySimpleGUI as sg
try:
    from .utils import now_str, date_str, view_playback
    from .utils import record as camera_record
    from .bag_utils import CompressionState
except:
    from tools.utils import now_str, date_str, view_playback 
    from tools.utils import record as camera_record
    from tools.bag_utils import CompressionState


# -------------Layouts-------------
BUTTONS = []
SEARCH = [[sg.I(size=(20, 1), key='-INPUT-',enable_events=True)],
          [sg.Listbox([], size=(20, 4), key='-LIST-',enable_events=True)],
          [sg.Push(),sg.B('Seleccionar',k = '-SELECT-PATIENT-')]]
PATIENT_FORM = [[sg.T('Nombre'),sg.I(key = '-NAME-',enable_events=True)],
           [sg.T('Edad'),sg.I(key = '-AGE-',s = 3),sg.T('Sexo'),sg.Push(),sg.Combo(['Hembra', 'Varon'],key = '-SEX-',default_value= 'Hembra')
           ,sg.Push(),sg.T('Color de Piel'),sg.Combo(['Blanco','Negro'],key = '-SKIN-',default_value= 'Blanco')],
           [sg.T('Tipo de Diabetes'), sg.Combo(['Tipo I', 'Tipo II', 'No diabetes'],key = '-DIABETES-TYPE-',default_value= 'Tipo I'),sg.Push(),
           sg.T('Tipo de Ulcera'),sg.Combo(['Neuroinfecciosa', 'Isquémica', 'Flebolinfática', 'Combinada'],key='-ULCER-TYPE-',default_value= 'Neuroinfecciosa')],
           [sg.Push(),sg.T('Localización'), sg.Combo(['Pie Derecho','Pie Izquierdo'],key = '-LOC-',default_value= 'Pie Derecho'),sg.Push(),
             sg.CalendarButton('Fecha', close_when_date_chosen=True,  target='-FIRST-DATE-', no_titlebar=False, 
                                                      month_names = ('Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'),
                                                      day_abbreviations=('Do','Lu','Ma','Mi','Ju','Vi','Sa'))
             ,sg.Input(key='-FIRST-DATE-', size=(20,1))],
           [sg.Push(),sg.B('Salvar')]]
DATE = [
    [sg.CalendarButton('Fecha', close_when_date_chosen=True,  target='-VIDEO-DATE-', no_titlebar=False, key = 'GET-DATE',
                                                      month_names = ('Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'),
                                                      day_abbreviations=('Do','Lu','Ma','Mi','Ju','Vi','Sa'))
    ,sg.Input(key='-VIDEO-DATE-', size=(20,1))],
    [sg.Push(),sg.Button('Comenzar Grabación'),sg.Push()]
]

COMPRESS = [
    [sg.Text('',key = '-COMPRESS-TEXT-')],
    [sg.ProgressBar(1000,key = '-COMRPESS-BAR-'),sg.Text('',key = '-COMPRESS-PERCENT-')]
    ]

# -----------Column Labels-----------
BUTTONS_LABEL = '-BUTTONS-'
SEARCH_LABEL = '-SEARCH-'
PATIENT_FORM_LABEL = '-PATIENT_FORM-'
DATE_LABEL = '-DATE-'
COMPRESS_LABEL = '-COMPRESS-'
# -------------Layout-------------
LAYOUT = [[sg.pin(sg.T('DFU-Measure')),sg.Push(),sg.pin(sg.B('Atrás', k = 'Prev',visible= False))],
          [sg.Column(BUTTONS,k = BUTTONS_LABEL,visible = True),
          sg.Column(SEARCH,k = SEARCH_LABEL,visible = False),
          sg.Column(PATIENT_FORM,k = PATIENT_FORM_LABEL,visible = False),
          sg.Column(DATE,k = DATE_LABEL,visible = False),
          sg.Column(COMPRESS,k = COMPRESS_LABEL,visible = False)]
        ]


class Gui:
    def __init__(self,layout) -> None:
        self.layout = layout
        self.current_window_label = None
        self.window = None
        self.button_count = 0

    def start(self):
        self.window = sg.Window('DFU-Measure',self.layout,use_custom_titlebar=True)
        self.window.read(1)
        self.window['-INPUT-'].update(disabled = True)
        self.window['-LIST-'].update(disabled = True)
        self.window['-NAME-'].update(disabled = True)
    

    def swap_layout(self,new_layout_label):
        if self.current_window_label:
            if self.current_window_label == new_layout_label:
                return
            self.window[self.current_window_label].update(visible = False)
        self.window[new_layout_label].update(visible = True)
        self.current_window_label = new_layout_label

    def filter(query:str,options:list[str]):
        return [item for item in options if query.upper() in item.upper()]




    def button_chose(self,options: list[str]):
        # Poner las opciones como botones
        self.swap_layout(BUTTONS_LABEL)
        # Crear button dict
        b_dict = dict([(f'-B-{index}-',key) for index, key in enumerate(options)])
        # Añadir botones en casos de faltar
        if self.button_count < len(options):
            for i in range(self.button_count,len(options)):
                self.window.extend_layout(self.window[BUTTONS_LABEL],[[sg.B(f'-B-{i}-',k = f'-B-{i}-',)]])
            self.button_count = len(options)
        # Poner todos los botones visibles
        for i in range(self.button_count):
            self.window[f'-B-{i}-'].update(visible = True)

        # Esconder botones en el caso de sobrare
        for i in range(len(options),self.button_count):
            self.window[f'-B-{i}-'].update(visible = False)
        
        for index, name in enumerate(options):
            self.window[f'-B-{index}-'].update(name)
        
        event, _ = self.window.read()
        try:
            return b_dict[event]
        except KeyError:
            return event
    


    def search_chose(self,options, initial_input = ''):
        # Buscar en un listado de nombres
        # Activando la ventana
        self.swap_layout(SEARCH_LABEL)
        self.window['-INPUT-'].update(disabled = False)
        self.window['-LIST-'].update(disabled = False)
        self.window['-INPUT-'].update(initial_input)
        self.window['-LIST-'].update(Gui.filter(initial_input,options))

        # Ciclo de eventos
        while True:
            event, values = self.window.read()
            if event in (sg.WIN_CLOSED, 'Exit','Prev'):  # always check for closed window
                return event, []
            if values['-INPUT-'] != '':  # if a keystroke entered in search field
                search = values['-INPUT-']
                new_values = Gui.filter(search,options)  # do the filtering
                
                self.window['-LIST-'].update(new_values)     # display in the listbox
            else:
                # display original unfiltered list
                self.window['-LIST-'].update(options)

            if event == '-INPUT-':
                continue
            # if a list item is chosen
            if event == '-LIST-' and len(values['-LIST-']):
                self.window['-INPUT-'].update(values['-LIST-'][0])
                new_values = Gui.filter( values['-LIST-'][0],options)  # do the filtering
                self.window['-LIST-'].update(new_values)   
            elif event == '-SELECT-PATIENT-':
                if not values['-INPUT-']:
                    continue
                self.window['-INPUT-'].update(disabled = True)
                self.window['-LIST-'].update(disabled = True)
                return event, values['-INPUT-']
            else:
                self.window['-INPUT-'].update(disabled = True)
                self.window['-LIST-'].update(disabled = True)
                return event, []

    def fix_name_format(name):
        if name[-1] == ' ':
            return ' '.join(item.capitalize() for item in name.split() if item) + ' '     
        else:
            return ' '.join(item.capitalize() for item in name.split() if item)



    def create_patient(self,valid_checker,patient = None):
        # Formulario para introducir los datos de un nuevo paciente
        self.swap_layout(PATIENT_FORM_LABEL)
        self.window['-NAME-'].update(disabled = False)
        #Setting Deffault values
        if not patient:
            self.window['-FIRST-DATE-'].update(now_str())
            self.window['-NAME-'].update('')
            self.window['-AGE-'].update('')
            self.window['-SEX-'].update('Hembra')
            self.window['-SKIN-'].update('Blanco')
            self.window['-DIABETES-TYPE-'].update('Tipo I')
            self.window['-ULCER-TYPE-'].update('Neuroinfecciosa')
            self.window['-LOC-'].update('Pie Derecho')
        else:
            self.window['-NAME-'].update(str(patient['Name']))
            self.window['-AGE-'].update(str(patient['Age']))
            self.window['-SEX-'].update(str(patient['Sex']))
            self.window['-SKIN-'].update(str(patient['Skin_Color']))
            self.window['-DIABETES-TYPE-'].update(str(patient['Diabetes_Type']))
            self.window['-ULCER-TYPE-'].update(str(patient['Ulcer_Type']))
            self.window['-LOC-'].update(str(patient['Localization']))
            self.window['-FIRST-DATE-'].update(date_str(patient['First_Date']))
        
        e = 'NOT-EMPTY'
        while(e):
            event, values = self.window.read()
            if event == '-NAME-':
                if values['-NAME-']:
                    self.window['-NAME-'].update(Gui.fix_name_format(values['-NAME-']))
            elif event == 'Salvar':
                data = (values['-NAME-'],
                        values['-AGE-'],
                        values['-SEX-'],
                        values['-SKIN-'],
                        values['-DIABETES-TYPE-'],
                        values['-ULCER-TYPE-'],
                        values['-LOC-'],
                        values['-FIRST-DATE-']
                        )
                e = valid_checker(data)
                if e:
                    sg.popup_error(e().message)
            elif not event:
                return event, []
            else:
                self.window['-NAME-'].update(disabled = True)
                return event, []
        self.window['-NAME-'].update(disabled = True)
        return event, data
        
    def get_date_for_video(self):
        self.swap_layout(DATE_LABEL)
        self.window['-VIDEO-DATE-'].update(now_str())
        while True:
            event, values = self.window.read()
            if event == 'Comenzar Grabación':
                return event, values['-VIDEO-DATE-']
            if not event or event == 'Prev' or event == 'Exit':
                return event, []

    def compress(self,compression_state: CompressionState, title):
        self.swap_layout(COMPRESS_LABEL)
        self.window['-COMPRESS-TEXT-'].update(title)
        while not compression_state.end:
            event, _ = self.window.read(.5)
            if not event or event == 'Prev' or event == 'Exit':
                return event
            current = compression_state.current_size
            fullsize = compression_state.full_size
            if current == None or fullsize == None:
                self.window['-COMRPESS-BAR-'].update(0)
                self.window['-COMPRESS-PERCENT-'].update(0)
            else:
                float_percent = 100*(current/fullsize)
                int_ave = int(1000*(current/fullsize))
                self.window['-COMRPESS-BAR-'].update(int_ave)
                self.window['-COMPRESS-PERCENT-'].update(('%.2f'%float_percent) + '%')
        return 'Ok'            


    def record(self,path):
        # Grabar la DFU de un paciete
        self.window.disappear()
        try:
            camera_record(path)
        except Exception:
            sg.popup_error('Conecta la cámara')
            self.window.reappear()
            return 'Prev'
        self.window.reappear()
        return 'Ok'

    def playback(self, video_path):
        # Reproducir una DFU
        self.window.disappear()
        view_playback(video_path)
        self.window.reappear()
        return 'Prev'
                

    def end(self):
        # Cerrar la ventana
        self.window.close()
        pass

    def set_back_visibility(self,visibility):
        # Hacer que se disponible el boton para retroceder del layout actual
        self.window['Prev'].update(visible = visibility)
        pass


        
