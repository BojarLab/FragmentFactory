import pandas as pd
import numpy as np
from CandyCrunch.prediction import bin_intensities

FF_data = pd.read_pickle('./FragmentFactory_dataset.pkl')
max_spectra_values = [max(d.values()) for d in FF_data['peak_d']]
FF_data['peak_d_norm'] = [{k:v for k,v in (pd.Series(d) / max_spectra_values[i]).items()} for i,d in enumerate(FF_data['peak_d'])]
FF_data['rounded_mass'] = [np.round(x,0) for x in FF_data['reducing_mass']]
frames = np.arange(40,1860,0.5)
binned_intensities, mz_remainder = zip(*[bin_intensities(c, frames) for c in FF_data['peak_d_norm']])
FF_data['binned_intensities_norm'] = binned_intensities
FF_data = FF_data.drop(columns=['peak_d_norm','rounded_mass'])
FF_data.to_pickle('./FragmentFactory_dataset_processed.pkl')
