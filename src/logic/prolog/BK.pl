:- use_module(library(apply)).
:- use_module(library(lists)).
% :- use_module(library(tabling)).
% :- table valid_rules/2, op_rule/2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DCG parser for equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% symbols to be mapped
digit(1).
digit(0).

% digits
digits([D]) --> [D], { digit(D) }. % empty list [] is not a digit
digits([D | T]) --> [D], !, digits(T), { digit(D) }.
digits(X):-
    phrase(digits(X), X),
    % This two clauses are only used when you want to prune the search space by
    % forbiding the first digit to be 0.
    length(X, L),
    (L > 1 -> X \= [0 | _]; true).

% Equation definition
eq_arg([D]) --> [D], { \+ D == '+', \+ D == '=' }.
eq_arg([D | T]) --> [D], !, eq_arg(T), { \+ D == '+', \+ D == '=' }.
equation(eq(X, Y, Z)) -->
    eq_arg(X), [+], eq_arg(Y), [=], eq_arg(Z),
    % rules for argument length
    { length(X, LX), length(Y, LY), length(Z, LZ),
      LZ =< max(LX, LY) + 1, LZ >= max(LX, LY) }.
parse_eq(List_of_Terms, Eq) :-
    phrase(equation(Eq), List_of_Terms).

%%%%%%%%%%%%%%%%%%%%%%
%% Bit-wise operation
%%%%%%%%%%%%%%%%%%%%%%
% Abductive calculation with given pseudo-labels, abduces pseudo-labels as well as operation rules
calc(Rules, Pseudo) :-
    calc([], Rules, Pseudo).
calc(Rules0, Rules1, Pseudo) :-
    parse_eq(Pseudo, eq(X,Y,Z)),
    bitwise_calc(Rules0, Rules1, X, Y, Z).

% Bit-wise calculation that handles carrying
bitwise_calc(Rules, Rules1, X, Y, Z) :-
    reverse(X, X1), reverse(Y, Y1), reverse(Z, Z1),
    bitwise_calc_r(Rules, Rules1, X1, Y1, Z1),
    maplist(digits, [X,Y,Z]).
bitwise_calc_r(Rs, Rs, [], Y, Y).
bitwise_calc_r(Rs, Rs, X, [], X).
bitwise_calc_r(Rules, Rules1, [D1 | X], [D2 | Y], [D3 | Z]) :-
    abduce_op_rule(my_op([D1],[D2],Sum), Rules, Rules2),
    ((Sum = [D3], Carry = []); (Sum = [C, D3], Carry = [C])),
    bitwise_calc_r(Rules2, Rules3, X, Carry, X_carried),
    bitwise_calc_r(Rules3, Rules1, X_carried, Y, Z).

%%%%%%%%%%%%%%%%%%%%%%%%%
% Abduce operation rules
%%%%%%%%%%%%%%%%%%%%%%%%%
% Get an existed rule
abduce_op_rule(R, Rules, Rules) :-
    member(R, Rules).
% Add a new rule
abduce_op_rule(R, Rules, [R|Rules]) :-
    op_rule(R),
    valid_rules(Rules, R).

% Integrity Constraints
valid_rules([], _).
valid_rules([my_op([X1],[Y1],_)|Rs], my_op([X],[Y],Z)) :-
    op_rule(my_op([X],[Y],Z)),
    [X,Y] \= [X1,Y1],
    [X,Y] \= [Y1,X1],
    valid_rules(Rs, my_op([X],[Y],Z)).
valid_rules([my_op([Y],[X],Z)|Rs], my_op([X],[Y],Z)) :-
    X \= Y,
    valid_rules(Rs, my_op([X],[Y],Z)).

op_rule(my_op([X],[Y],[Z])) :- digit(X), digit(Y), digit(Z).
op_rule(my_op([X],[Y],[Z1,Z2])) :- digit(X), digit(Y), digits([Z1,Z2]).
