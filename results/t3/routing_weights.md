## Overall

|         |   w_clean |   w_salt_pepper |   w_blur |   w_occlusion |
|:--------|----------:|----------------:|---------:|--------------:|
| overall |    0.1052 |             0.3 |   0.3013 |        0.2936 |

## By condition

| condition   |   w_clean |   w_salt_pepper |   w_blur |   w_occlusion |
|:------------|----------:|----------------:|---------:|--------------:|
| clean       |    0.9814 |               0 |   0.0125 |        0.0061 |
| salt_pepper |    0      |               1 |   0      |        0      |
| blur        |    0.0004 |               0 |   0.9996 |        0      |
| occlusion   |    0.023  |               0 |   0.0005 |        0.9765 |

## By condition and severity

|                           |   w_clean |   w_salt_pepper |   w_blur |   w_occlusion |
|:--------------------------|----------:|----------------:|---------:|--------------:|
| ('clean', 'none')         |    0.9814 |               0 |   0.0125 |        0.0061 |
| ('salt_pepper', 'low')    |    0      |               1 |   0      |        0      |
| ('salt_pepper', 'medium') |    0      |               1 |   0      |        0      |
| ('salt_pepper', 'high')   |    0      |               1 |   0      |        0      |
| ('blur', 'low')           |    0.0008 |               0 |   0.9991 |        0.0001 |
| ('blur', 'medium')        |    0.0002 |               0 |   0.9998 |        0      |
| ('blur', 'high')          |    0.0002 |               0 |   0.9998 |        0      |
| ('occlusion', 'low')      |    0.0682 |               0 |   0.0014 |        0.9304 |
| ('occlusion', 'medium')   |    0.0009 |               0 |   0      |        0.9991 |
| ('occlusion', 'high')     |    0      |               0 |   0      |        1      |
