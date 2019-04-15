# UnDeco
Unsupervised Systematics Removal

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


## Usage

The package comes with two classes:

```
PCA and SYSREM
```

The first one is the Principal Component Analisys, while the second is SYSREM (Tamuz et al. 2005):

```
sys = SYSREM(pre_processing=True, norm_method='normalize', after_processing=False)
pca = PCA(pre_processing=True, norm_method='normalize', after_processing=False)

result_sys = sys.fit_transform(matrx, components=1)
result_pca = pca.fit_transform(matrx, components=1, comp_end=(-1))
```

The ```components``` argument is the number of components with higher variance to be substracted.
The ```comp_end``` argument is the number of components with lower variance to be subtracted.
The ```after_processing``` divide each column of the result matrix by its standard deviation.


## Version

1.0.0

## Authors

* **Mario Damiano** - Github: [MDamiano](https://github.com/MDamiano) - Twitter: [@astromariodam](https://twitter.com/astromariodam) - e-mail: mdamiano91@gmail.com

## License

This project is distributed under the MIT License - see the [LICENSE](https://github.com/MDamiano/UnDeco/blob/master/LICENSE) file for details
