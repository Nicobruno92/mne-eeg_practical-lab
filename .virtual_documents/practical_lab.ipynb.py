### Importamos las librerias que vamos a usar ###

#esto es para que los gráficos aparezcan fuera de jupyter ya que los gráficos de MNE que son interactivos
get_ipython().run_line_magic("matplotlib", " qt ")

# Algunas librerías que podrían resultar de utilidad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# importar mne
import mne
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
# import openneuro #to install it pip install openneuro-py




# direccion de la carpeta donde descargamos el archivo
# por ejemplo '/Users/nicobruno/Downloads/practical_lab/'
# Download one subject's data from each dataset
bids_root = '/Users/nicobruno/Downloads/practical_lab'


print(make_report(bids_root))


# Solo nos va a interesar la data de EEG
datatype = 'eeg'
suffix = 'eeg'

#vamoss a analizar la task GO-NOGO
task = 'gonogo' 

# los sujetos tenian 2 runs por task pero solo vamos a usar el 1
run = '1'

#Subject that we want to use
# We are going to start with 'sub-AB4'
subject = 'AB12'

bids_path = BIDSPath(subject=subject, task=task, run = run,
                     suffix=suffix, datatype=datatype, 
                    root=bids_root)


display(bids_path)


# leer el archivo crudo pasando el path que acabamos de crear al archivo
raw = read_raw_bids(bids_path=bids_path, verbose = False)


display(raw.info) #display es como print pero se ve mas lindo en jupyter


raw.info['sfreq']  # Para ver la frequencia de sampleo especificamente


raw.info['bads']  # para ver si algun canal esta marcado como malo


raw.ch_names[:10] # el nombre de los primeros 10 canales


len(raw.info['chs']) #cuantos canales hay


raw.load_data()  # it is required to load data in memory
# Downsamplear la data a la mitad
raw.resample(250) 


raw.plot()


# definir valor high pass filter
hpass = 1
# definir valor low pass filter
lpass = 40

# filtrar data cruda
# buscar en la documentacion
# van a tener que utilizar el metodo de mne para filtrar la data 
raw_filtered = raw.filter(hpass,lpass)


raw_filtered.plot_psd()


# usar funcion events_from_annotations
events, event_id = mne.events_from_annotations(raw = raw_filtered)


print(event_id)
print(events[:10]) #solo los 10 primeros eventos


# visualizar los eventos
mne.viz.plot_events(events, raw_filtered.info['sfreq'])


# lista con los id de los eventos de interes 
include = [5, 6]

# events to include using pick events
events_included = mne.pick_events(events = events, include = [5,6])

# diccionario de eventos a incluir
event_id_included  ={'go': 5, 
                    'no-go': 6}


# definir tiempo minimo
tmin = -0.2
# definir tiempo maximo
tmax = 0.8
# definir baseline
baseline = (-0.2,0)


#criterio de rejecteo predifinido
reject_criteria = dict(eeg= 200e-6, eog=200e-6) 


epochs = mne.Epochs(raw_filtered, events = events_included, event_id = event_id_included, 
                    tmin = tmin, tmax = tmax, baseline=baseline, reject = reject_criteria) 


print(epochs)


epochs.plot()


epochs.drop_bad()  # remover las epocas malas
epochs.plot_drop_log() #plotear que épocas dropeamos y bajo que condiciones


print(epochs) # ahora deberiamos tener tantas epocas menos como las que dropeamos


# NO SE COMO GUARDAR EN BIDS. SI NO SE PPUEDE DE BAJAAA


n_components = 0.99  # Should normally be higher, like 0.999get_ipython().getoutput("!")
method = 'fastica'
max_iter = 512  # Should normally be higher, like 500 or even 1000get_ipython().getoutput("!")
fit_params = dict(fastica_it=5)
random_state = 42

ica = mne.preprocessing.ICA(n_components=n_components,
                            method=method,
                            max_iter=max_iter,
#                             fit_params=fit_params,
                            random_state=random_state)

ica.fit(epochs)


ica.plot_components(inst = epoch,picks=range(10))


ica.plot_sources(epochs, block=False, picks =range(10))


# epochs interpolated
epochs_interpolate = epochs.interpolate_bads()


# usar average para la referencia de los canales
epochs_rereferenced, ref_data = mne.set_eeg_reference(inst = epochs_interpolate, ref_channels = 'average', copy = True)



