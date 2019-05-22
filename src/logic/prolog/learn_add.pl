:- ensure_loaded(['BK.pl']).
:- thread_setconcurrency(_, 8).
:- use_module(library(thread)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For propositionalisation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eval_inst_feature(Ex, Feature):-
    eval_eq(Ex, Feature).

%% Evaluate instance given feature
eval_eq(Ex, Feature):-
    parse_eq(Ex, eq(X,Y,Z)),
    bitwise_calc(Feature,_,X,Y,Z), !.

%%%%%%%%%%%%%%
%% Abduction
%%%%%%%%%%%%%%
% Make abduction when given examples that have been interpreted as pseudo-labels
abduce(Exs, Delta_C) :-
    abduce(Exs, [], Delta_C).
abduce([], Delta_C, Delta_C).
abduce([E|Exs], Delta_C0, Delta_C1) :-
    calc(Delta_C0, Delta_C2, E),
    abduce(Exs, Delta_C2, Delta_C1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Abduce pseudo-labels only
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
abduce_consistent_insts(Exs):-
    abduce(Exs, _), !.
% (Experimental) Uncomment to use parallel abduction
% abduce_consistent_exs_concurrent(Exs), !.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Abduce Delta_C given pseudo-labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
consistent_inst_feature(Exs, Delta_C):-
    abduce(Exs, Delta_C), !.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (Experimental) Parallel abduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
abduce_consistent_exs_concurrent(Exs) :-
    % Split the current data batch into grounding examples and variable examples (which need to be revised)
    split_exs(Exs, Ground_Exs, Var_Exs),
    % Find the simplest Delta_C for grounding examples.
    abduce(Ground_Exs, Ground_Delta_C), !,
    % Extend Ground Delta_C into all possible variations
    extend_op_rule(Ground_Delta_C, Possible_Deltas),
    % Concurrently abduce the variable examples
    maplist(append([abduce2, Var_Exs, Ground_Exs]), [[Possible_Deltas]], Call_List),
    maplist(=.., Goals, Call_List),
    % writeln(Goals),
    first_solution(Var_Exs, Goals, [local(inf)]).

split_exs([],[],[]).
split_exs([E | Exs], [E | G_Exs], V_Exs):-
    ground(E), !,
    split_exs(Exs, G_Exs, V_Exs).
split_exs([E | Exs], G_Exs, [E | V_Exs]):-
    split_exs(Exs, G_Exs, V_Exs).

:- table extend_op_rule/2.

extend_op_rule(Rules, Rules) :-
    length(Rules, 4).
extend_op_rule(Rules, Ext) :-
    op_rule(R),
    valid_rules(Rules, R),
    extend_op_rule([R|Rules], Ext).

% abduction without learning new Delta_C (Because they have been extended!)
abduce2([], _, _).
abduce2([E|Exs], Ground_Exs, Delta_C) :-
    % abduce by finding ground examples
    member(E, Ground_Exs),
    abduce2(Exs, Ground_Exs, Delta_C).
abduce2([E|Exs], Ground_Exs, Delta_C) :-
    eval_inst_feature(E, Delta_C),
    abduce2(Exs, Ground_Exs, Delta_C).
