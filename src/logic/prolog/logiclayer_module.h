#ifndef Py_LOGICLAYERMODULE_H
#define Py_LOGICLAYERMODULE_H

#include <SWI-cpp.h>
#include <string>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>

using namespace std;

/****** Module methods ******/
static PyObject *
LogicLayer_initProlog(PyObject * self, PyObject *args); // initialise prolog engine

static PyObject *
LogicLayer_haltProlog(PyObject * self, PyObject *args); // destroy prolog engine

static PyObject *
LogicLayer_consultKB(PyObject * self, PyObject *args); // consult a knowledge base

static PyObject *
LogicLayer_gc(PyObject * self, PyObject *args); // call prolog garbage collect

static PyObject *
LogicLayer_trimStacks(PyObject * self, PyObject *args); // call prolog to trim stack memory

static PyObject *
LogicLayer_call(PyObject * self, PyObject *args); // call prolog command

static PyObject *
LogicLayer_legitInst(PyObject * self, PyObject *args); // test if the (mapped) example is legitimate

static PyObject *
LogicLayer_genRandFeature(PyObject * self, PyObject *args); // generate random feature

static PyObject *
LogicLayer_parseInstFeature(PyObject * self, PyObject *args); // parse an instance as feature

static PyObject *
LogicLayer_evalInstFeature(PyObject * self, PyObject *args); // evaluate instance with feature return true/false (1/0)

static PyObject *
LogicLayer_evalInstRules(PyObject * self, PyObject *args); // evaluate instance with a list of my_op rules and return true/false (1/0)

static PyObject *
LogicLayer_abduceInstFeature(PyObject * self, PyObject *args); // abduce instance mapping given feature and label

static PyObject *
LogicLayer_abduceConInsts(PyObject * self, PyObject *args); // abduce consistent instances

static PyObject *
LogicLayer_conInstsFeature(PyObject * self, PyObject *args); // transform consistent instances to LL feature

static PyObject *
LogicLayer_conDigitRules(PyObject * self, PyObject *args); // test if digit rules are consistent

/* List of functions defined in the module */
static PyMethodDef LogicLayer_methods[] = {
        { "init", LogicLayer_initProlog, METH_VARARGS,
          PyDoc_STR("init() -> Initialise Prolog engine.") },
        { "halt", LogicLayer_haltProlog, METH_VARARGS,
          PyDoc_STR("halt() -> Halt Prolog engine.") },
        { "gc", LogicLayer_gc, METH_VARARGS,
          PyDoc_STR("gc() -> Call Prolog garbage collect.") },
        { "trimStacks", LogicLayer_trimStacks, METH_VARARGS,
          PyDoc_STR("trimStacks() -> Call Prolog trim stack.") },
        { "consult", LogicLayer_consultKB, METH_VARARGS,
          PyDoc_STR("consult(File) -> Consult a knowledge base.") },
        { "call", LogicLayer_call, METH_VARARGS,
          PyDoc_STR("call(Atom) -> Call Prolog atom.") },
        { "legitInst", LogicLayer_legitInst, METH_VARARGS,
          PyDoc_STR("legitInst(Inst) -> Evaluate if the mapped instance is legitimate according to Knowledge Base.") },
        { "genRandFeature", LogicLayer_genRandFeature, METH_VARARGS,
          PyDoc_STR("genRandFeature() -> Generate a random logical feature.") },
        { "parseInstFeature", LogicLayer_parseInstFeature, METH_VARARGS,
          PyDoc_STR("LogicLayer_parseInstFeature(Inst) -> parse an instance as feature.") },
        { "evalInstFeature", LogicLayer_evalInstFeature, METH_VARARGS,
          PyDoc_STR("evalInstFeature(Inst, Feature) -> 0/1. Evaluate instance given a logical feature, 0 for false, 1 for true.") },
        { "evalInstRules", LogicLayer_evalInstRules, METH_VARARGS,
          PyDoc_STR("evalInstRules(Inst, Rules) -> 0/1. Evaluate instance given a set of my_op rules, 0 for false, 1 for true.") },
        { "abduceInstFeature", LogicLayer_abduceInstFeature, METH_VARARGS,
          PyDoc_STR("abduceInstFeature(Inst, Feature, Label) -> Mapping. Abduce a mapping given logical feature and label.") },
        { "abduceConInsts", LogicLayer_abduceConInsts, METH_VARARGS,
          PyDoc_STR("abduceConInsts(Inst, Label) -> Abduce a consistent binding of a set of examples if possible, else return None") },
        { "conInstsFeature", LogicLayer_conInstsFeature, METH_VARARGS,
          PyDoc_STR("conInstsFeature(Inst, Label) -> Transform consistent instances to LL feature, if not possible then return None") },
        { "conDigitRules", LogicLayer_conDigitRules, METH_VARARGS,
          PyDoc_STR("conDigitRules(Rules) -> test if digit rules are consistent") },
        { NULL, NULL, 0, NULL }           /* sentinel */
};

PyDoc_STRVAR(module_doc, "A Layer for Logical Abduction.");
        
static struct PyModuleDef LogicLayerModule = {
    PyModuleDef_HEAD_INIT,
    "LogicLayer",
    module_doc,
    0,
    LogicLayer_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

/**** Py_PlTerm Object ****/

typedef struct {
        PyObject_HEAD
        PyObject * term_py; // do not store PlTerm in memory
} Py_PlTerm;

#define Py_PlTerm_Check(v)      (Py_TYPE(v) == &Py_PlTerm_Type)

static PyObject *
Py_PlTerm_new(PyTypeObject *type, PyObject *args, PyObject *kwds); // new

static int
Py_PlTerm_init(Py_PlTerm *self, PyObject *args, PyObject *kwds); // init

static PlTerm
Py2PlTerm(PyObject *atom, map<string, PlTerm> *var_map); // recursively analyse a prolog term from python object

static PyObject *
PlTerm2Py(PlTerm term); // recursively convert prolog term to python object

static void
Py_PlTerm_dealloc(Py_PlTerm *self); // destructor

static PyObject *
Py_PlTerm_repr(Py_PlTerm *self); // representation

static Py_PlTerm *
Py_PlTerm_fromPlTerm(PlTerm term); // create Py_PlTerm from prolog term

static PyObject *
Py_PlTerm_toPy(Py_PlTerm *term); // cast prolog term to python object

PyDoc_STRVAR(Py_PlTerm_doc,
"PlTerm(x) -> Prolog Term \n\
\n\
Convert a string to a Prolog Term, if possible.");

static PyMethodDef Py_PlTerm_methods[] = {
    { "py", (PyCFunction) Py_PlTerm_toPy, METH_NOARGS,
      PyDoc_STR("py() -> Cast prolog term to python object.") },
    { NULL, NULL, 0, NULL }           // sentinel
};

static PyTypeObject Py_PlTerm_Type = {
        // The ob_type field must be initialized in the module init function
        // to be portable to Windows without using C++.
        PyVarObject_HEAD_INIT(&PyType_Type, 0)
        "LL.PlTerm",                          //tp_name
        sizeof(Py_PlTerm),                    //tp_basicsize
        0,                                    //tp_itemsize
        // methods
        (destructor)Py_PlTerm_dealloc,        //tp_dealloc
        0,                                    //tp_print
        (getattrfunc)0,                       //tp_getattr
        0,                                    //tp_setattr
        0,                                    //tp_reserved
        (reprfunc)Py_PlTerm_repr,             //tp_repr
        0,                                    //tp_as_number
        0,                                    //tp_as_sequence
        0,                                    //tp_as_mapping
        0,                                    //tp_hash
        0,                                    //tp_call
        (reprfunc)Py_PlTerm_repr,             //tp_str
        PyObject_GenericGetAttr,              //tp_getattro
        0,                                    //tp_setattro
        0,                                    //tp_as_buffer
        Py_TPFLAGS_DEFAULT,                   //tp_flags
        Py_PlTerm_doc,                        //tp_doc
        0,                                    //tp_traverse
        0,                                    //tp_clear
        0,                                    //tp_richcompare
        0,                                    //tp_weaklistoffset
        0,                                    //tp_iter
        0,                                    //tp_iternext
        Py_PlTerm_methods,                    //tp_methods
        0,                                    //tp_members
        0,                                    //tp_getset
        0,                                    //tp_base
        0,                                    //tp_dict
        0,                                    //tp_descr_get
        0,                                    //tp_descr_set
        0,                                    //tp_dictoffset
        (initproc)Py_PlTerm_init,             //tp_init
        0,                                    //tp_alloc
        Py_PlTerm_new,                        //tp_new
        0,                                    //tp_free
        0,                                    //tp_is_gc
};

/* Export function for the module (*must* be called PyInit_xx) */

PyMODINIT_FUNC
PyInit_LogicLayer(void) {
    PyObject* m;

    if (PyType_Ready(&Py_PlTerm_Type) < 0)
        return NULL;
    
    m = PyModule_Create(&LogicLayerModule);
    if (m == NULL)
        return NULL;
    
    Py_INCREF(&Py_PlTerm_Type);
    PyModule_AddObject(m, "PlTerm", (PyObject *)&Py_PlTerm_Type);
    return m;
}

/*
  
// Logical Layer Object

// TEMPORARILY ABOLISHED, SINCE ONLY ONE PROLOG ENGINE IS ALLOWED PER PROCESS.
// IN THE FUTURE, WE WILL USE PROLOG MODULE (NOT PYTHON MODULE) TO IMPLEMENT
// THE LOGICAL LAYER OBJECT, WHICH WILL ENABLE USING MULTIPLE INDEPENDENT LOGIC
// LAYERS WITH DIFFERENT BACKGROUND KNOWLEDGE BASES FOR NEURAL LOGICAL NETWORK.

typedef struct {
        PyObject_HEAD
        PyObject    *ID;             // Layer ID
        PyObject    *KB_Path;        // Path of knowledge base
} LogicLayerObject;

#define LogicLayerObject_Check(v)      (Py_TYPE(v) == &LogicLayerObject_Type)

// Logical Layer methods, implemented in .c files

static LogicLayerObject *
newLogicLayerObject(PyObject *arg); // constructor

static void
LogicLayerObject_dealloc(LogicLayerObject *self); // destructor

static PyObject *
LogicLayerObject_getKBPath(LogicLayerObject *self, PyObject *args); // get Knowledge Base path

static PyObject *
LogicLayerObject_setKB(LogicLayerObject *self, PyObject *args); // set KB

static PyObject *
LogicLayerObject_test(LogicLayerObject *self, PyObject *args); // testing

// List of Methods
static PyMethodDef LogicLayerObject_methods[] = {
        { "test", (PyCFunction) LogicLayerObject_test, METH_VARARGS,
          PyDoc_STR("test() -> test logic engine and return none.") },
        { "getKBPath", (PyCFunction) LogicLayerObject_getKBPath, METH_VARARGS,
          PyDoc_STR("KBPath() -> get Knowledge Base path.") },
        { "setKB", (PyCFunction) LogicLayerObject_setKB, METH_VARARGS,
          PyDoc_STR("setKB() -> set Knowledge Base and consult it.") },
        { NULL, NULL, 0, NULL }           // sentinel
};

// Object Type definition

static PyTypeObject LogicLayerObject_Type = {
        // The ob_type field must be initialized in the module init function
        // to be portable to Windows without using C++.
        PyVarObject_HEAD_INIT(NULL, 0)
        "LogicLayer.LogicLayer",              //tp_name
        sizeof(LogicLayerObject),             //tp_basicsize
        0,                                    //tp_itemsize
        // methods
        (destructor)LogicLayerObject_dealloc, //tp_dealloc
        0,                                    //tp_print
        0,                                    //tp_getattr
        0,                                    //tp_setattr
        0,                                    //tp_reserved
        0,                                    //tp_repr
        0,                                    //tp_as_number
        0,                                    //tp_as_sequence
        0,                                    //tp_as_mapping
        0,                                    //tp_hash
        0,                                    //tp_call
        0,                                    //tp_str
        PyObject_GenericGetAttr,              //tp_getattro
        0,                                    //tp_setattro
        0,                                    //tp_as_buffer
        Py_TPFLAGS_DEFAULT,                   //tp_flags
        0,                                    //tp_doc
        0,                                    //tp_traverse
        0,                                    //tp_clear
        0,                                    //tp_richcompare
        0,                                    //tp_weaklistoffset
        0,                                    //tp_iter
        0,                                    //tp_iternext
        LogicLayerObject_methods,             //tp_methods
        0,                                    //tp_members
        0,                                    //tp_getset
        0,                                    //tp_base
        0,                                    //tp_dict
        0,                                    //tp_descr_get
        0,                                    //tp_descr_set
        0,                                    //tp_dictoffset
        0,                                    //tp_init
        0,                                    //tp_alloc
        0,                                    //tp_new
        0,                                    //tp_free
        0,                                    //tp_is_gc
};
*/

#ifdef __cplusplus
}
#endif

#endif /* !defined(Py_LOGICLAYERMODULE_H) */
