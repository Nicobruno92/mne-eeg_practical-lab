get_ipython().run_line_magic("matplotlib", " qt ")


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report


# Completa con el path que lleve a tus datos
bids_root = '...'


print(make_report(bids_root))


# Solo nos va a interesar la data de EEG
datatype = 'eeg'
suffix = 'eeg'

# Vamos a analizar la task GO-NOGO
task = 'gonogo' 

# Los sujetos tenian 2 _runs_ por task pero solo vamos a usar el 1
run = '1'

# Sujeto que queremos utilizar
# Vamos a empezar utilizando el 'sub-AB12'
subject = 'AB12'

bids_path = BIDSPath(subject=subject, task=task, run = run,
                     suffix=suffix, datatype=datatype, 
                    root=bids_root)


display(bids_path) # display es como print pero se ve más lindo en jupyter :)


# Leer el archivo crudo pasando el path que acabamos de crear al archivo
raw = ...


# Elimino canales que no son de EEG
raw = raw.drop_channels(['M1', 'M2', 'PO5', 'PO6', 'CB1', 'CB2', 'R-Dia-X-(mm)', 'R-Dia-Y-(mm)'])
# Seteo el montage de acuerdo a las condiciones experimentales (en el trabajo usaron Biosemi de 64 canales)
raw.set_montage('biosemi64')
# Plot del montage
raw.plot_sensors(show_names=True)


display(raw.info) 


raw.info['sfreq']  # Para ver la frequencia de sampleo (i.e. cantidad de registros por segundo) 


raw.info['...']  # Para ver si algun canal fue marcado como "malo" (i.e. defectuoso) por parte de los investigadores


# Ver el nombre de los primeros 10 canales
...


#Ver cuántos cuantos canales hay
...


# Carga de datos a memroia (requeria para algunas operaciones)
raw.load_data()  
# Downsamplear la data a la mitad
...


raw.plot()


# definir valor high pass filter
hpass = ...
# definir valor low pass filter
lpass = ...

# filtrar data cruda
# buscar en la documentacion
# van a tener que utilizar el metodo de mne para filtrar la data 
raw_filtered = raw.filter(..., ...)


raw_filtered.plot_psd()


# usar funcion events_from_annotations
events, event_id = ...


print(events[:10]) #solo los 10 primeros eventos
print(event_id)


# visualizar los eventos
mne.viz.plot_events(events, raw_filtered.info['sfreq'])


# lista con los id de los eventos de interés ("Go" y "No-Go") 
include = ...

# events to include using pick events
events_included = mne.pick_events(...)

# diccionario de eventos a incluir
event_id_included  = ...


# definir tiempo minimo
tmin = ...
# definir tiempo maximo
tmax = ...
# definir baseline
baseline = ...


#criterio de rejecteo predifinido
reject_criteria = dict(eeg= 200e-6, eog=200e-6) 


epochs = ...


print(epochs)


# Plotear las épocas
...


epochs.drop_bad()  # remover las epocas malas
epochs.plot_drop_log() #plotear que épocas dropeamos y bajo que condiciones


print(epochs) # ahora deberiamos tener tantas epocas menos como las que dropeamos


epochs_fname = bids_path.copy().update(suffix='epo', check=False)

# Utilizar el método save para guardar épocas
...


# Correr directamente este bloque de código que permite identificar componentes que pudieran ser artefactos a través del método ICA

n_components = 0.99  
method = 'fastica'
max_iter = 512 
fit_params = dict(fastica_it=5)
random_state = 42

ica = mne.preprocessing.ICA(n_components=n_components,
                            method=method,
                            max_iter=max_iter,
                            random_state=random_state)

ica.fit(epochs)


ica.plot_sources(epochs, block=False, picks =range(10))


# epochs interpolated
epochs.load_data()
epochs_interpolate = ...


# usar average para la referencia de los canales
epochs_rereferenced, ref_data = mne.set_eeg_reference(inst = ..., ref_channels = '...', copy = True)


# Cómputo de Evoked para la condición Go
epochs_rereferenced["go"].average().plot(spatial_colors=True)


# Cómputo de Evoked para la condición No-Go
...


average_go = epochs_rereferenced["go"].average()
average_no_go = ...
# Comparamos ERPs
...


frontal = ...
posterior = ...

evokeds = dict(Go=list(epochs_rereferenced['go'].iter_evoked()),
               ...))


# Calculamos el Evoked para electrodos frontales



# Calculamos el Evoked para electrodos posteriores
