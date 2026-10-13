# Dataset report

## Sources (as read)

| Source | Count |
|---|---|
| Images with captions | 1200 |
| Caption strings | 6000 |
| Label rows (`Training_set.csv`) | 6499 |
| Image files on disk | 1200 |

## Combined CSV

| Check | Value |
|---|---|
| Data rows | 6000 |
| Unique sentence ids | 6000 |
| Duplicate sentence ids | 0 |
| Source captions missing from CSV | 0 |
| Label mismatches | 0 |
| Repeated caption strings | 470 |
| Templated 'A photo of a ...' captions | 1200 |

## Coverage

| Check | Value |
|---|---|
| Captioned images without a label | 0 |
| Labelled images without captions | 0 |
| Captioned images with no file on disk | 0 |
| Image files with no CSV row | 0 |
| Files not matching Image_<n>.<ext> | 0 |
| Repeated image indices on disk | 0 |

## Extras

- **images**: 1200
- **species**: 75
- **split_images**: {'train': 957, 'val': 137, 'test': 106}
- **split_captions**: {'train': 4785, 'val': 685, 'test': 530}
- **split_species**: {'train': 75, 'val': 75, 'test': 75}
- **dataset_fingerprint**: 05a5380e0b185eb18b4ad9ebc04fb5df54417305

