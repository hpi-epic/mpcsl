# parallel-pc

## Contributing

### Installing libraries

To build from this source you need to have the
[GSL](https://www.gnu.org/software/gsl/) and
[Armadillo](http://arma.sourceforge.net/) installed.

Please feel free to add instructions for your system, especially when it's more
complicated than `brew install ...`.

#### On MacOS (using [homebrew](https://brew.sh/))

To install these dependencies on MacOS using homebrew run:

```sh
brew install gsl
brew install armadillo
```
#### On UNIX

To install these dependencies on Unix run:

```sh
sudo apt install libgsl-dev
sudo apt install libarmadillo-dev
```

### Other dependencies

#### Submodules

This project uses the
[concurrentqueue](https://github.com/cameron314/concurrentqueue/tree/8f7e861dd9411a0bf77a6b9de83a47b3424fafba) project
internally. After cloning the repository for the first time, run
`git submodule update --init --recursive` to pull into your working directory as well.
