# LogicLayer Module #

This is a logic abduction module for python.

## Build ##

1. First change the `swipl_include_dir` and `swipl_lib_dir` in `setup.py` to your own SWI-Prolog path.

2. Build
```shell
cd src/prolog
python3 setup.py install
```
## Usage ##

```python
# initialisation
import LogicLayer as LL
LL.init()
LL.consult('PATH_TO_KNOWLEDGE_BASE')
```

More examples please refer to `test.py`

## Known Bugs ##

1. If logic engine has been halted in a process, it cannot be initialised again, i.e., `LogicLayer.init()` cannot be invoked after `LogicLayer.halt()`. This is a known bug of SWI-Prolog.

2. If the background file path is wrong, `LogicLayer.consult()` will stuck. Will be fixed by catching this exception.

## To Be Done ##

1. Concurrent inference and abduction by implementing prolog modules. (Cannot open concurrent prolog engine due to the constraint of SWI-Prolog). However, using modules might require lot of memory.

