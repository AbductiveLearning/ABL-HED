#include <SWI-cpp.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <map>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

#include "logiclayer_module.h"

#include <Python.h>

using namespace std;

/** Function for module **/
// initialise the main prolog engine
static PyObject *
LogicLayer_initProlog(PyObject * self, PyObject *args) {
    char *stack = NULL;
    //if (!PyArg_ParseTuple(args, "s:init", &stack))
    if (!PyArg_ParseTuple(args, "|s:ref", &stack))
        return NULL;

    if (!stack)
        stack = (char *) "";

    int ac = 0;
    char **av = (char **)malloc(sizeof(char *) * (20));
    char av0[10] = "./";
    char av1[10] = "-q";         // quiet
    char av2[15] = "-nosignals"; // signal handling
    av[ac++] = av0;
    av[ac++] = av1;
    av[ac++] = av2;

    const char *split = " ";
    char *p; 
    p = strtok(stack, split);
    while (p != NULL) {
        av[ac++] = p;
        p = strtok(NULL, split);
    }
    av[ac] = NULL;

    if (!PL_is_initialised(NULL, NULL))
        if (!PL_initialise(ac, av))
            cerr << "Initialise Prolog Engine failed!" << endl;

    Py_INCREF(Py_None);
    free(av);
    return Py_None;
}

// halt the main prolog engine
static PyObject *
LogicLayer_haltProlog(PyObject * self, PyObject *args) {

    if (!PyArg_ParseTuple(args, ":halt"))
        return NULL;

    if (PL_is_initialised(NULL, NULL))
        PL_cleanup(1);
    else
        cerr << "No Prolog engine exists!" << endl;

    Py_INCREF(Py_None);
    return Py_None;
}

// call Prolog garbage collect
static PyObject *
LogicLayer_gc(PyObject * self, PyObject *args) {
    if (!PyArg_ParseTuple(args, ":GC"))
        return NULL;

    fid_t fid = PL_open_foreign_frame();
    if (PL_is_initialised(NULL, NULL)) {
        char gc[] = "garbage_collect";
        if (!PlCall(gc))
            cerr << "Call garbage_collect/0 failed!" << endl;
    } else
        cerr << "No Prolog engine exists!" << endl;
    
    PL_discard_foreign_frame(fid);
    Py_INCREF(Py_None);
    return Py_None;
}

// call prolog to trim stack memory
static PyObject *
LogicLayer_trimStacks(PyObject * self, PyObject *args) {
    if (!PyArg_ParseTuple(args, ":trimStack"))
        return NULL;

    fid_t fid = PL_open_foreign_frame();
    if (PL_is_initialised(NULL, NULL)) {
        char ts[] = "trim_stacks";
        if (!PlCall(ts))
            cerr << "Call trim_stack/0 failed!" << endl;
    } else
        cerr << "No Prolog engine exists!" << endl;
    
    PL_discard_foreign_frame(fid);
    Py_INCREF(Py_None);
    return Py_None;

}


// consult a knowledge base file
static PyObject *
LogicLayer_consultKB(PyObject * self, PyObject *args) {
    char *path;
    if (!PyArg_ParseTuple(args, "s:consult", &path)) {
        cerr << "Wrong file path inputted!" << endl;
        return NULL;
    }

    fid_t fid = PL_open_foreign_frame();

    char query[256];
    sprintf(query, "consult('%s')", path);
    if (!PlCall(query))
        cerr << "Consult file failed!" << endl;

    PL_discard_foreign_frame(fid);
    Py_INCREF(Py_None);
    return Py_None;
}

// call prolog atom
static PyObject *
LogicLayer_call(PyObject * self, PyObject *args) {
    char *atom;
    if (!PyArg_ParseTuple(args, "s:call", &atom)) {
        cerr << "Wrong input!" << endl;
        return NULL;
    }
    fid_t fid = PL_open_foreign_frame();
    char query[256];
    sprintf(query, "(%s)", atom);
    if (!PlCall(query))
        cerr << "Calling failed!" << endl;
    PL_discard_foreign_frame(fid);
    Py_INCREF(Py_None);
    return Py_None;
}

// test if the (mapped) example is legitimate
static PyObject *
LogicLayer_legitInst(PyObject * self, PyObject *args) {
    PyObject *ex_; // input
    if (!PyArg_ParseTuple(args, "O:legitInst", &ex_)) {
        cerr << "Wrong input!" << endl;
        return NULL;
    }
    fid_t fid = PL_open_foreign_frame();
    Py_PlTerm *ex = (Py_PlTerm *) ex_;
    
    PlTermv av(1);
    map<string, PlTerm> *var_map = new map<string, PlTerm>();
    av[0] = Py2PlTerm(ex->term_py, var_map);
    delete var_map;

    predicate_t pred = PL_predicate("legitimate_ex", 1, "user");
    qid_t q = PL_open_query(NULL, PL_Q_NORMAL, pred, av.a0);
    int result = PL_next_solution(q) ? 1 : 0;
    PL_close_query(q);
    PL_reset_term_refs(av.a0);
    
    PyObject *ans = PyBool_FromLong(result);
    PL_discard_foreign_frame(fid);
    Py_INCREF(ans);
    return (PyObject *) ans;
}

// generate random feature
static PyObject *
LogicLayer_genRandFeature(PyObject * self, PyObject *args) {
    if (!PyArg_ParseTuple(args, ":genRandFeature")) {
        return NULL;
    }
    fid_t fid = PL_open_foreign_frame();
    PlTermv av(1);
    av[0] = PlTerm();

    predicate_t pred = PL_predicate("gen_random_feature", 1, "user");
    qid_t q = PL_open_query(NULL, PL_Q_NORMAL, pred, av.a0);
    if (!PL_next_solution(q)) {
        PL_close_query(q);
        cerr << "Random feature generation failed!" << endl;

        PL_reset_term_refs(av.a0);
        PL_discard_foreign_frame(fid);
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        PL_cut_query(q);
        //PL_discard_foreign_frame(fid);
        
        Py_PlTerm *re;
        re = Py_PlTerm_fromPlTerm(av[0]);

        PL_reset_term_refs(av.a0);
        PL_discard_foreign_frame(fid);
        Py_INCREF(re);
        return (PyObject *) re;
    }
}

// parse an instance as feature
static PyObject *
LogicLayer_parseInstFeature(PyObject * self, PyObject *args) {
    PyObject *ex_; // input
    if (!PyArg_ParseTuple(args, "O:parseInstFeature", &ex_)) {
        cerr << "Wrong input!" << endl;
        return NULL;
    }
    fid_t fid = PL_open_foreign_frame();

    PlTermv av(2);
    map<string, PlTerm> *var_map = new map<string, PlTerm>();
    
    av[1] = PlTerm();
    if (Py_PlTerm_Check(ex_))
        av[0] = Py2PlTerm(((Py_PlTerm *) ex_)->term_py, var_map);
    else
        av[0] = Py2PlTerm(ex_, var_map);

    delete var_map;
    
    predicate_t pred = PL_predicate("parse_feature", 2, "user");
    qid_t q = PL_open_query(NULL, PL_Q_NORMAL, pred, av.a0);
   
    if (PL_next_solution(q)) {
        PL_cut_query(q);
        Py_PlTerm *re = Py_PlTerm_fromPlTerm(av[1]);
        PL_reset_term_refs(av.a0);
        PL_discard_foreign_frame(fid);
        Py_INCREF(re);
        return (PyObject *) re;
    } else {
        PL_close_query(q);
        PL_reset_term_refs(av.a0);
        PL_discard_foreign_frame(fid);
        cerr << "Cannot parse feature from this example: ";
        Py_INCREF(Py_None);
        return Py_None;
    }
}

// evaluate instance with feature return true/false (1/0)
static PyObject *
LogicLayer_evalInstFeature(PyObject * self, PyObject *args) {
    PyObject *ex_, *feat_; // input
    if (!PyArg_ParseTuple(args, "OO:evalInstFeature", &ex_, &feat_)) {
        return NULL;
    }
    Py_PlTerm *ex = (Py_PlTerm *) ex_, *feat = (Py_PlTerm *) feat_;
    fid_t fid = PL_open_foreign_frame();
    PlTermv av(2);
    map<string, PlTerm> *var_map1 = new map<string, PlTerm>();
    map<string, PlTerm> *var_map2 = new map<string, PlTerm>();
    av[0] = Py2PlTerm(ex->term_py, var_map1);
    av[1] = Py2PlTerm(feat->term_py, var_map2);
    delete var_map1;
    delete var_map2;
    
    predicate_t pred = PL_predicate("eval_inst_feature", 2, "user");
    qid_t q = PL_open_query(NULL, PL_Q_NORMAL, pred, av.a0);
    int result = PL_next_solution(q) ? 1 : 0;
    PL_close_query(q);
    PL_reset_term_refs(av.a0);
    PL_discard_foreign_frame(fid);
    PyObject *ans = PyBool_FromLong(result);
    Py_INCREF(ans);
    return (PyObject *) ans;
}

// evaluate instance with ruleset return true/false (1/0)
static PyObject *
LogicLayer_evalInstRules(PyObject * self, PyObject *args) {
    PyObject *ex_, *feat_; // input
    if (!PyArg_ParseTuple(args, "OO:evalInstRules", &ex_, &feat_)) {
        return NULL;
    }
    PyObject *feat = (PyObject *) feat_;

    fid_t fid = PL_open_foreign_frame();
    PlTermv av(2);
    map<string, PlTerm> *var_map = new map<string, PlTerm>();
    if (Py_PlTerm_Check(ex_))
        av[0] = Py2PlTerm(((Py_PlTerm *) ex_)->term_py, var_map);
    else
        av[0] = Py2PlTerm(ex_, var_map);

    av[1] = Py2PlTerm(feat, var_map);
    delete var_map;

    predicate_t pred = PL_predicate("eval_inst_feature", 2, "user");
    qid_t q = PL_open_query(NULL, PL_Q_NORMAL, pred, av.a0);
    int result = PL_next_solution(q) ? 1 : 0;
    PL_close_query(q);
    PL_reset_term_refs(av.a0);
    PL_discard_foreign_frame(fid);
    PyObject *ans = PyBool_FromLong(result);
    Py_INCREF(ans);
    return (PyObject *) ans;
}

// abduce instance mapping given feature and label
static PyObject *
LogicLayer_abduceInstFeature(PyObject * self, PyObject *args) {
    PyObject *ex_, *feat_, *label; // input
    if (!PyArg_ParseTuple(args, "OOO:abduceInstFeature",
                          &ex_, &feat_, &label)) {
        return NULL;
    }
    Py_PlTerm *ex = (Py_PlTerm *) ex_;
    Py_PlTerm *feat = (Py_PlTerm *) feat_;

    fid_t fid = PL_open_foreign_frame();
    map<string, PlTerm> *var_map1 = new map<string, PlTerm>();
    map<string, PlTerm> *var_map2 = new map<string, PlTerm>();
    map<string, PlTerm> *var_map3 = new map<string, PlTerm>();

    PlTermv av(3);
    av[0] = Py2PlTerm(ex->term_py, var_map1);
    av[1] = Py2PlTerm(feat->term_py, var_map2);
    av[2] = Py2PlTerm(label, var_map3);
    delete var_map1;
    delete var_map2;
    delete var_map3;
    
    PyObject *re = PyList_New(0);

    predicate_t pred = PL_predicate("abduce_inst_feature", 3, "user");
    qid_t q = PL_open_query(NULL, PL_Q_NORMAL, pred, av.a0);

    while (PL_next_solution(q)) {
        Py_PlTerm *t = Py_PlTerm_fromPlTerm(av[0]);
        Py_INCREF(t);
        PyList_Append(re, (PyObject *) t);
    }
    PL_close_query(q);
    PL_reset_term_refs(av.a0);
    PL_discard_foreign_frame(fid);

    Py_INCREF(re);
    return re;
}

// abduce consistent instances
static PyObject *
LogicLayer_abduceConInsts(PyObject * self, PyObject *args) {
    PyObject *ex_; // input
    if (!PyArg_ParseTuple(args, "O:abduceConInsts", &ex_)) {
        cerr << "Wrong input!" << endl;
        return NULL;
    }

    fid_t fid = PL_open_foreign_frame();
    PlTerm ex;
    map<string, PlTerm> *var_map = new map<string, PlTerm>();
    ex = Py2PlTerm(ex_, var_map);
    delete var_map;
    
    predicate_t pred = PL_predicate("abduce_consistent_insts", 1, "user");
    qid_t q = PL_open_query(NULL, PL_Q_NORMAL, pred, ex.ref); // use c interface

    PyObject *re;

    if (PL_next_solution(q)) {
        // only get the first consistent abduction
        re = (PyObject *) Py_PlTerm_fromPlTerm(ex);
        PL_close_query(q);
    } else {
        re = Py_None;
        PL_close_query(q);
    }
    PL_reset_term_refs(ex.ref);
    PL_discard_foreign_frame(fid);
    
    Py_INCREF(re);
    return re;
}

// transform consistent instances to LL feature
static PyObject *
LogicLayer_conInstsFeature(PyObject * self, PyObject *args) {
    PyObject *ex_; // input
    if (!PyArg_ParseTuple(args, "O:conInstsFeature", &ex_)) {
        cerr << "Wrong input!" << endl;
        return NULL;
    }

    fid_t fid = PL_open_foreign_frame();
    PlTermv av(2);
    map<string, PlTerm> *var_map = new map<string, PlTerm>();
    av[0] = Py2PlTerm(ex_, var_map);
    av[1] = PlTerm();
    delete var_map;

    //cout << (char *) av[0] << endl;
    
    predicate_t pred = PL_predicate("consistent_inst_feature", 2, "user");
    qid_t q = PL_open_query(NULL, PL_Q_NORMAL, pred, av.a0); // use c interface

    PyObject *re;

    if (PL_next_solution(q)) {
        // only get the first consistent abduction
        re = (PyObject *) Py_PlTerm_fromPlTerm(av[1]);
        PL_close_query(q);
    } else {
        re = Py_None;
        PL_close_query(q);
    }
    PL_reset_term_refs(av.a0);
    PL_discard_foreign_frame(fid);
    
    Py_INCREF(re);
    return re;    
}

static PyObject *
LogicLayer_conDigitRules(PyObject * self, PyObject *args) {
    PyObject *ex_; // input
    if (!PyArg_ParseTuple(args, "O:conDigitRules", &ex_)) {
        cerr << "Wrong input!" << endl;
        return NULL;
    }

    fid_t fid = PL_open_foreign_frame();
    PlTermv av(1);
    map<string, PlTerm> *var_map = new map<string, PlTerm>();
    av[0] = Py2PlTerm(ex_, var_map);
    delete var_map;

    predicate_t pred = PL_predicate("consistent_digit_rules", 1, "user");
    qid_t q = PL_open_query(NULL, PL_Q_NORMAL, pred, av.a0); // use c interface
    int result = PL_next_solution(q) ? 1 : 0;
    PL_close_query(q);
    PL_reset_term_refs(av.a0);
    PL_discard_foreign_frame(fid);
    PyObject *ans = PyBool_FromLong(result);
    Py_INCREF(ans);
    return (PyObject *) ans;
}

/********** Py_PlTerm Methods *********/
// recursively analyse a prolog term from PyObject
static PlTerm 
Py2PlTerm(PyObject *atom, map<string, PlTerm> *var_map) {
    PlTerm re; // returned term
    if (PyList_Check(atom)) {
        // if the input is a list, do list analyse
        term_t tail_ref = PL_new_term_ref();
        PlTerm tail_term(tail_ref);
        PlTail pl_list(tail_term);
        long len = (long) PyList_Size(atom);
        for (int i = 0; i < len; i++) {
            PlTerm elem = Py2PlTerm(PyList_GetItem(atom, i), var_map);
            pl_list.append(elem);
        }
        pl_list.close();
        re = tail_term;
    } else {
        // do compound analyse
        if (Py_PlTerm_Check(atom)) { // if is already a Py_PlTerm
            Py_INCREF(atom);
            map<string, PlTerm> *var_map_t = new map<string, PlTerm>(); 
            re = Py2PlTerm(((Py_PlTerm *) atom)->term_py, var_map_t);
            delete var_map_t;
            //cout << "Org Py_PlTerm: " << (char *) *t << endl;
            //cout << "re: " << (char *) re << endl;
        }
        if (PyBool_Check(atom)) { // if is bool
            if (atom == Py_True)
                re = PlTerm("true");
            else if (atom == Py_False)
                re = PlTerm("false");
            else
                cerr << "Parsing boolean object failed." << endl;
        } else if (PyLong_Check(atom)) {
            re = PlTerm(PyLong_AsLong(atom));
        } else if (PyFloat_Check(atom)) {
            re = PlTerm(PyFloat_AsDouble(atom));
        } else if (PyUnicode_Check(atom)) {
            string word(PyBytes_AsString(PyUnicode_AsUTF8String(atom)));
            if ((word.size() >= 1 && std::isupper(word[0]))
                || (word.size() > 1 && word[0] == '_')) { // named var
                auto iter = var_map->find(word);
                if (iter != var_map->end()) { // if the var has appeared before
                    re = iter->second;
                } else {
                    re = PlTerm();
                    var_map->insert(pair<string, PlTerm>(word, re));
                }
            } else {
                PlCompound c(word.c_str());
                re = (PlTerm) c;
            }
        }
    }
    //cout << (char *) re << " TYPE ---> " << re.type() << endl;
    return re;
}

static int
Py_PlTerm_init(Py_PlTerm *self, PyObject *args, PyObject *kwds) {
    PyObject *term = NULL;
    PyObject *tmp_term = NULL;
    
    static char *kwlist[] = {(char *) "term", 0};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:PlTerm", kwlist, &term))
        return -1;
    if (term) {
        tmp_term = self->term_py;
        Py_INCREF(term);
        self->term_py = term; // allocate new term
        Py_XDECREF(tmp_term);
    }

    return 0;
}

static PyObject *
Py_PlTerm_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Py_PlTerm *self;
    self = PyObject_New(Py_PlTerm, &Py_PlTerm_Type);

    if (self == NULL) {
        return NULL;
    }        
    self->term_py = NULL;
    return (PyObject *) self;
}

// destructor
static void
Py_PlTerm_dealloc(Py_PlTerm *self) {
    Py_XDECREF(self->term_py);
    PyObject_Del(self);
}

// construct Py_PlTerm (a PyObject wrapper) from prolog PlTerm
static Py_PlTerm *
Py_PlTerm_fromPlTerm(PlTerm term) {
    Py_PlTerm *self;
    self = PyObject_New(Py_PlTerm, &Py_PlTerm_Type);
    if (self == NULL)
        return NULL;
    self->term_py = PlTerm2Py(term);
    if (self->term_py == NULL) {
        Py_DECREF(self);
        cerr << "Allocating memory for Py_PlTerm failed!" << endl;
        return NULL;
    }
    return self;
}

// representation
static PyObject *
Py_PlTerm_repr(Py_PlTerm *self) {
    return PyObject_Repr(self->term_py);
}

// recursively convert prolog term to python object
static PyObject *
PlTerm2Py(PlTerm term) {
    PyObject *re;

    switch (term.type()) {
    case PL_NIL:
        re = PyList_New(0);
        if (re == NULL)
            return NULL;
        
        break;
    case PL_LIST_PAIR:
    {
        //cout << (char *) term << " is a list." << endl;
        PlTail tail(term);
        PlTerm t;

        re = PyList_New(0);
        if (re == NULL)
            return NULL;
        while(tail.next(t)) {
            PyObject *e = PlTerm2Py(t);
            if (PyList_Append(re, e) < 0)
                return NULL;
        }

        break;
    }
    case PL_ATOM:
        //cout << (char *) term << " is an atom." << endl;
        if (strcmp((char *) term, "true") == 0) {
            re = Py_True;
        } else if ((strcmp((char *) term, "false")) == 0) {
            re = Py_False;
        } else {
            re = PyUnicode_FromString((char *) term);
        }
        if (re == NULL)
            return NULL;
        
        break;
    case PL_INTEGER:
        //cout << (char *) term << " is an int." << endl;
        re = PyLong_FromLong((long) term);
        if (re == NULL)
            return NULL;

        break;
    case PL_FLOAT:
        //cout << (char *) term << " is a float." << endl;
        re = PyFloat_FromDouble((double) term);
        if (re == NULL)
            return NULL;

        break;
    case PL_VARIABLE:
        //cout << (char *) term << " is a variable." << endl;
        re = PyUnicode_FromString((char *) term);
        if (re == NULL)
            return NULL;

        break;
    default:
        //cout << (char *) term << " is an default." << endl;
        re = PyUnicode_FromString((char *) term);
        if (re == NULL)
            return NULL;
        
    }

    Py_INCREF(re);
    return re;
}

static PyObject *
Py_PlTerm_toPy(Py_PlTerm *self) {
    return self->term_py;
}

#ifdef __cplusplus
}
#endif
