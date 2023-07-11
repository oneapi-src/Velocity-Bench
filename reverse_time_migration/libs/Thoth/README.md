# I/O Framework (Thoth)

## Table of content
- [Description](#description)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Hierarchy](#project-hierarchy)
- [Versioning](#versioning)
- [Changelog](#changelog)
- [License](#license)


## Description
General I/O library for parsing, manipulating, creating, editing and visualizing different types of seismic file formats for developers to be used in different seismic applications.


## Prerequisites
* **CMake**\
Cmake version 3.5 or higher.

* **C++**\
C++14 standard supported compiler.

* **Catch2**\
Already included in the repository in ```prerequisites/catch```

## Features

* Under construction

## Project Hierarchy
* **```prerequisites```**\
Folder containing the prerequisites needed for the project, or default scripts to install them.

* **```include```**\
The folder containing all the headers of the system. Contains ReadMe explaining the internal file structure of the project.

* **```src```**\
The folder containing all the source files of the system. Follows same structure as the include.

* **```tests```**\
The folder containing all the tests of the system. Follows same structure as the include.

* **```examples```**\
The folder containing all the demo code showcasing how the framework is used within applications, and the capabilities of the framework.

* **```clean_build.sh```**\
Script used to build the system tests after running the config.sh, and by default build all the different modules of the project.

* **```config.sh```**\
Script used to generate the building system inside a 'bin' directory.

* **```CMakeLists.txt```**\
The top level CMake file to configure the build system.

## Versioning

When installing I/O Framework (Thoth), require it's version. For us, this is what ```major.minor.patch``` means:

- ```major``` - **MAJOR breaking changes**; includes major new features, major changes in how the whole system works, and complete rewrites; it allows us to _considerably_ improve the product, and add features that were previously impossible.
- ```minor``` - **MINOR breaking changes**; it allows us to add big new features.
- ```patch``` - **NO breaking changes**; includes bug fixes and non-breaking new features.


## Changelog

For previous versions, please see our [CHANGELOG](CHANGELOG.rst) file.


## License

This project is licensed under the The GNU Lesser General Public License, version 3.0 (LGPL-3.0) Legal License - see the [LICENSE](LICENSE.txt) file for details
