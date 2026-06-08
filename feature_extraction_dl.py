```python
import librosa
import numpy as np

def extract_feature_dl(file_path):

    y, sr = librosa.load(
        file_path,
        sr=None
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    max_len = 200

    if mfcc.shape[1] < max_len:

        mfcc = np.pad(

            mfcc,

            ((0,0),
             (0,max_len-mfcc.shape[1])),

            mode='constant'
        )

    else:

        mfcc = mfcc[:, :max_len]

    return mfcc.T
```
