import pandas as pd
import numpy as np
from CandyCrunch.prediction import bin_intensities

FF_data = pd.read_pickle('./FragmentFactory_dataset.pkl')
max_spectra_values = [max(d.values()) for d in FF_data['peak_d']]
FF_data['peak_d_norm'] = [{k:v for k,v in (pd.Series(d) / max_spectra_values[i]).items()} for i,d in enumerate(FF_data['peak_d'])]
FF_data['rounded_mass'] = [np.round(x,0) for x in FF_data['reducing_mass']]
for mass_value in FF_data['rounded_mass'].unique():
    new_frames  = np.arange(40,mass_value+5,0.5)
    FF_data.loc[FF_data['reducing_mass'].round(0) == mass_value,'binned_intensities_norm'] = FF_data.loc[FF_data['reducing_mass'].round(0) == mass_value,'peak_d_norm'].apply(lambda x: bin_intensities(x, new_frames)[0])
FF_data = FF_data.drop(columns=['peak_d_norm','rounded_mass'])
FF_data.to_pickle('./FragmentFactory_dataset_processed.pkl')