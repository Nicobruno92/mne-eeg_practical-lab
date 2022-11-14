get_ipython().run_line_magic("matplotlib", " qt ")


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report


bids_root = '/Users/nicobruno/Downloads/practical_lab'


print(make_report(bids_root))


# Solo nos va a interesar la data de EEG
datatype = 'eeg'
suffix = 'eeg'

# Nombre que le vamos a poenr a la tarea (GO-NOGO)
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
raw = read_raw_bids(bids_path=bids_path, verbose = False)


# Elimino canales que no son de EEG
raw = raw.drop_channels(['M1', 'M2', 'PO5', 'PO6', 'CB1', 'CB2', 'R-Dia-X-(mm)', 'R-Dia-Y-(mm)'])
# Seteo el montage de acuerdo a las condiciones experimentales (en el trabajo usaron Biosemi de 64 canales)
raw.set_montage('biosemi64')
raw.plot_sensors(show_names=True)


display(raw.info) 


raw.info['sfreq']  # Para ver la frequencia de sampleo (i.e. cantidad de registros por segundo) 


raw.info['bads']  # Para ver si algun canal fue marcado como "malo" (i.e. defectuoso) por parte de los investigadores


raw.ch_names[:10] # el nombre de los primeros 10 canales


len(raw.info['chs']) #cuantos canales hay


# Carga de datos a memroia (requeria para algunas operaciones)
raw.load_data()  
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


print(events[:10]) #solo los 10 primeros eventos
print(event_id)


# visualizar los eventos
mne.viz.plot_events(events, raw_filtered.info['sfreq'])


# lista con los id de los eventos de interés ("Go" y "No-Go") 
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


epochs_fname = bids_path.copy().update(suffix='epo', check=False)
epochs.save(epochs_fname, overwrite=True)


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


ica.plot_components(inst = epochs, picks=range(10))


ica.plot_sources(epochs, block=False, picks =range(10))


ica.apply(epochs)


# epochs interpolated
epochs.load_data()
epochs_interpolate = epochs.interpolate_bads()


# usar average para la referencia de los canales
epochs_rereferenced, ref_data = mne.set_eeg_reference(inst = epochs_interpolate, ref_channels = 'average', copy = True)


epochs_rereferenced["go"].average().plot(spatial_colors=True)


epochs_rereferenced["no-go"].average().plot(spatial_colors=True)


average_go = epochs_rereferenced["go"].average()
average_no_go = epochs_rereferenced["no-go"].average()
mne.viz.plot_compare_evokeds(dict(Go=average_go, NoGo=average_no_go),
                             legend='upper left', show_sensors='upper right')


frontal = ['F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2']
posterior = ['P1', 'Pz', 'P2', 'PO3', 'POz', 'PO4']

evokeds = dict(Go=list(epochs_rereferenced['go'].iter_evoked()),
               NoGo=list(epochs_rereferenced['no-go'].iter_evoked()))


mne.viz.plot_compare_evokeds(evokeds, picks=frontal, combine='mean')


mne.viz.plot_compare_evokeds(evokeds, picks=posterior, combine='mean')


from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler, Vectorizer, cross_val_multiscore)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_auc_score


X = epochs_rereferenced.get_data()  # EEG signals: n_epochs, n_eeg_channels, n_times
y = epochs_rereferenced.events[:, 2]  # target: go vs nogo


clf = make_pipeline(
    Scaler(epochs_rereferenced.info, scalings='mean'),
    Vectorizer(),
    RandomForestClassifier(random_state = 42)
)


scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=1, scoring = 'roc_auc')

# Mean scores across cross-validation splits
score = np.mean(scores, axis=0)
print('Clasificacion AUC: get_ipython().run_line_magic("0.1f%%'", " % (100 * score,))")


# Resolver ejercicio de Test de Permutaciones


time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc', verbose=True)

# Acá usamos cv=3 por una cuestión de velocidad. 
scores_cv = cross_val_multiscore(time_decod, X, y, cv=3, n_jobs=1)

# Calculamos la media a través de las CrossValidation scores
scores = np.mean(scores_cv, axis=0)


# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
plt.show()


# define the Temporal generalization object
time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc',
                                 verbose=True)

# again, cv=3 just for speed
scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=1)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(epochs.times, np.diag(scores), label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')


fig, ax = plt.subplots(1, 1)
im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
               extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Temporal generalization')
ax.axvline(0, color='k')
ax.axhline(0, color='k')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('AUC')
plt.show()
