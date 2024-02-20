from utils.SAM_utils import create_sam_inter, review_images, TYPE_VIT_B, WPATH_VIT_B
from pathlib import Path
'''
Usar las flechas para navegar por las imágenes
Usar espacio para crear el embedding y poder hacer el prompt
    - click izquierdo para poner un promtp de foreground
    - click derecho para poner un prompt de background
    - 'C' para borrar el prompt
    - 'ENTER' para hacer la segmentación
    - 'Escape' para volver a poder navegar en las fotos
'''

data_path = Path('output\\color')
path_list = [str(path) for path in data_path.rglob('*.jpg')]
path_list = path_list[0:15]
review_images(path_list,create_sam_inter(TYPE_VIT_B,WPATH_VIT_B))