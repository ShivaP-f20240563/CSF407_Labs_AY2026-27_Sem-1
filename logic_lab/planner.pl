% Prolog as an independent plan verifier (Tasks 8-10).
% Run:  swipl -q -s planner.pl -g run_queries -t halt

% ---- Task 8: warehouse connectivity (facts) ----
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

% Rule: Connected(X,Y) -> CanMove(X,Y)
can_move(X,Y) :- connected(X,Y).

% ---- Task 9: checking proposed moves ----
valid_move(X,Y) :- connected(X,Y).

% Extension: a whole movement plan is valid if it starts at Start
% and every consecutive move is a valid_move from the current location.
valid_plan(_, []).
valid_plan(At, [move(At,Y)|Rest]) :- valid_move(At,Y), valid_plan(Y, Rest).

% ---- Task 10: forward-chaining style rules ----
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.

% ---- query runner ----
q(G) :- ( catch(G, E, (print_message(error, E), fail)) -> R = true ; R = false ),
        format("?- ~q.~t~40|~w~n", [G, R]).

run_queries :-
    q(can_move(a,b)), q(can_move(a,c)),
    q(valid_move(a,b)), q(valid_move(b,c)), q(valid_move(a,c)),
    q(valid_plan(a, [move(a,b), move(b,c)])),
    q(valid_plan(a, [move(a,c)])),
    q(reduce_speed).
