# botMaker

The domain language of this repository: a from-scratch ML library whose newest area is certified
landmark selection for the Nyström approximation by A* search. Terms are grouped by area; each entry
says what the thing *is*. Bracketed keys cite `docs/REFERENCES.md`. Implementation lives in code and
`docs/ARCHITECTURE.md`, decisions in `docs/DECISIONS.md`, techniques in `docs/LEARNING_LOG.md`.

## Language

### Search and certification

General search vocabulary, in its standard computer-science sense. The Nyström problem is one
instance; its own terms are in the next section.

**State**:
A configuration of the problem that the search can be in; the vertex of the (implicit or
materialised) graph being searched. Two paths that reach the same configuration reach the same
state. Here, a state is a set of chosen landmarks.
_Avoid_: node (except in `nodes_expanded`, an existing API name), position

**Goal state**:
A state that is a complete solution, at which the search may stop.

**Successor**:
A state reachable from another by one action. A problem defines its successors; the search does
not.

**Parent / child**:
A state, and a successor of it, seen from the search's side: the search expands parents and
prices children.

**Expansion**:
Taking the best state off the frontier and generating its successors. The number of expansions is
the standard measure of a search's cost.
_Avoid_: pop, visit

**Frontier**:
The set of generated states not yet expanded, ordered by their bound.
_Avoid_: fringe, open list

**Completion**:
Any goal state reachable from a given state.

**Objective**:
The value a search minimises, defined on goal states. In this repository it is a property of the
completed solution alone, never of the path to it.
_Avoid_: cost (kept for the cost-contract's names), error

**Lower bound**:
A value that is never above the objective of any completion of a state. A bound with this property
is admissible; a search ordered by an admissible bound proves that the first goal it takes off the
frontier is optimal. At a goal state the bound equals the objective. [L3]
_Avoid_: heuristic, f-value

**Certificate**:
The proof of optimality that an admissible search delivers when it terminates within its budget.
Without termination there is no certificate, only a solution. [A1]

**Bounded certificate**:
A proof that a solution is within a stated factor of optimal, as delivered by a weighted search
variant that trades exactness for fewer expansions. [A2]

**Brute force**:
Evaluating every candidate solution to obtain the ground truth.
_Avoid_: exhaustive, enumeration

**Optimum set**:
All solutions that attain the optimal objective. Ties are real; a certified answer is checked for
membership in this set, never for equality with one member.

**Oracle**:
A reference implementation, simple and slow, that any faster implementation of the same value is
tested against.

**Tie collapse**:
The regime where many states share the same bound, so an admissible search must expand nearly all
of them and the certificate degenerates into enumeration.

**Expanded set**:
The states already expanded, kept so a state reached twice is not expanded twice.
_Avoid_: closed set, explored set

**Frontier peak**:
The largest number of states the frontier held during a search; the search's memory cost, as
expansions are its time cost.

**Incumbent**:
The smallest objective of any goal state seen so far; an upper bound on the optimum. It may be seeded
from any complete solution before the search starts.
_Avoid_: best-so-far, current best

**Seed**:
A value that sets something's initial state before it runs, in every domain here: a random
generator's seed, a search's incumbent seed, a subsample's row seed. Always name what is seeded.

**Goal sibling**:
One of the goal states priced from the same parent in one expansion. Only the one with the smallest
bound can be the first goal expanded, so it is the only one the frontier needs.

**Pruning**:
Declining to place a priced child on the frontier because its bound exceeds an upper bound on the
optimum. The upper bound is the incumbent, or a quantity derived from an untruncated lower bound.
_Avoid_: filtering, culling

**Memory cap**:
A run stopped because its frontier or resident memory exceeded a budget. It has a solution only if a
goal state was seen, and never a certificate.

### Nyström approximation and column subset selection

**Landmark**:
A chosen data index whose kernel column represents the others in the Nyström approximation. [L2]
_Avoid_: column (unless the object is literally a column of the kernel square root), pivot, inducing point

**Residual kernel**:
What the landmarks leave unexplained: the kernel minus its Nyström reconstruction, a Schur
complement of the kernel with respect to the landmark block.
_Avoid_: error matrix

**Residual trace**:
The trace of the residual kernel; the objective every selector minimises and every result reports.
_Avoid_: error, cost (kept for the search contract's names), residual energy

**Column subset selection**:
Choosing columns of a matrix so that projecting onto their span loses the least Frobenius energy.
Landmark selection is column subset selection applied to the kernel's square root. [A1]

**Goal depth**:
The depth at which a child is a complete selection, so its bound is its true residual trace.

**Remaining energy**:
The residual trace a partial selection leaves; the first term of the lower bound.

**Best possible completion**:
The most residual trace that any further choice of landmarks could remove, as bounded by the top
eigenvalues of what remains; the term subtracted from the remaining energy. [A1]
_Avoid_: heuristic term, h

**Explained column**:
A column already in the span of the chosen landmarks; adding it changes nothing.

**Rank-k floor**:
The residual of the best rank-k approximation, which no set of k landmarks can beat. [B3]
_Avoid_: SVD residual, optimal rank-k error

**Subset gap**:
The best subset's residual trace over the rank-k floor: the price of choosing landmarks at all. [A4, B3]
_Avoid_: gap 2

**Selection gap**:
A selection rule's residual trace over the best subset's: the price of that rule. [B1]
_Avoid_: gap 1, optimality gap

**Bandwidth scale**:
The multiplier applied to the median-distance bandwidth when building a kernel; the dial from
narrow (fast-decaying spectrum) to wide (flat spectrum).
_Avoid_: gamma scale (kept as the parameter's name)

### Spectra and numerics

**Root decomposition**:
The eigendecomposition of the kernel done once before the search; everything the bound needs is
derived from it.

**Reduced coordinates**:
The kernel's square root expressed in the eigenbasis, so that all column geometry is preserved in
as many dimensions as the retained rank.

**Numeric rank**:
The number of eigenvalues above the relative tolerance that separates signal from rounding.

**Retained rank**:
The number of eigenvalues the bound actually uses; equal to the numeric rank unless the spectrum is
truncated.

**Dropped mass**:
The sum of the eigenvalues a truncation discards, expressed as a fraction of the trace.

**Spectrum truncation**:
Computing the bound on the kernel with its smallest eigenvalues dropped. The bound stays admissible
because the residual trace can only shrink when the kernel does; goal costs are never truncated.

**Parent spectrum / child spectrum**:
The eigenvalues of what remains after a parent's landmarks, and after one more landmark.

**Rank-one downdate**:
Obtaining a child spectrum from the parent spectrum by subtracting a single outer product, solved
through the secular equation instead of a fresh decomposition. [A1, A3, L4]

**Secular equation**:
The scalar equation whose roots are the eigenvalues of a rank-one modification of a diagonal
matrix; each root lies in an interval fixed by interlacing. [L4]

**Interlacing**:
The property that each downdated eigenvalue sits between two consecutive parent eigenvalues.

**Clamp**:
Holding a value that is non-negative in exact arithmetic at zero when cancellation rounds it below.
Which side of zero rounding lands on is platform-dependent and is never asserted.

### Baselines and reporting

**Published baseline**:
A selection rule from the literature that a result is compared against: greedy trace [L1], pivoted
Cholesky, RPCholesky [B1], uniform sampling.

**Instrumented heuristic**:
A rule that exists only to probe the search, such as greedy on the lower bound; never reported as
a baseline.

**Greedy trace**:
The rule that adds the landmark removing the most residual trace at each step. [L1]
_Avoid_: greedy Nyström, greedy gain

**Pivoted Cholesky**:
The rule that adds the landmark with the largest residual diagonal at each step.

**RPCholesky**:
Randomly pivoted Cholesky: the rule that samples the next landmark in proportion to the residual
diagonal. [B1]

**Seed summary**:
The mean, median, minimum and maximum of a randomised rule over independent seeds; the only way a
randomised rule is quoted.

**Best-of-N**:
The best single draw among N random selections; always named with its N.
_Avoid_: random (for anything but a single labelled draw)

**Cell**:
One experimental configuration: dataset, kernel, bandwidth scale, n and k, with every selector's
result recorded against the certified optimum.

### Two-hot spanning sets

The relaxation side of graph partitioning: a continuous optimizer moves a spanning set whose columns
are rounded to vertex pairs. Senses follow the research glossary
(`method/16-relaxation-landscape.md`) and `docs/plans/2026-09-two-hot-span.md`.

**2-hot vector**:
A vector with exactly two non-zero coordinates, of opposite sign: a vertex pair written as a vector.
_Avoid_: two-sparse vector, indicator

**Spanning set**:
The n×r matrix V whose columns the optimizer moves, rewarded for being 2-hot. Its span, not its
individual columns, is what the relaxed objective sees.
_Avoid_: basis (the columns need not be independent), embedding

**Collision measure**:
The per-column regularizer R(v_j) that rewards a column for concentrating on two coordinates. A
diagnostic of how 2-hot a column has become, never the criterion a run is judged by.
_Avoid_: sparsity penalty, concentration loss

**Roundability**:
How far a relaxed column is from its nearest 2-hot vector, and hence how much the rounded cut can
exceed the relaxed objective.
_Avoid_: rounding error, discretization gap

**Rounded pair**:
The two coordinates a column is rounded to: its largest and its smallest entry, the vertex pair the
column stands for.
_Avoid_: support, top-two

**Rounded cut**:
Ê, the ratio-cut value of the partition the rounded pairs induce; the value the relaxation actually
delivers, as against the value it relaxed to.
_Avoid_: discrete cut, final cut

**Relaxed objective**:
E\*, the objective at the unrounded V, computed through the exact `pinv` projector (D-31) and never
through the ridge projector the training loss uses.
_Avoid_: training loss (a different quantity), relaxed cut

**Projector**:
P_V, the projection onto the span of a spanning set, read through the residual it leaves on X. One
concept with two arithmetics: the exact pseudo-inverse projector that every reported number comes
through, and the ridge projector, smooth in a training knob ε, that the training loss descends.
_Avoid_: projection matrix, ridge alone (the knob, not the projector)

**Spectral floor**:
Σλ, the sum of the K smallest Laplacian eigenvalues: the lower bound every E\* and Ê is read
against.
_Avoid_: floor alone where the rank-k floor could be meant

**Component count**:
The number of connected components the rounded pairs induce, compared with the K clusters asked for.
_Avoid_: cluster count (K is what was asked; this is what was got)

### Composition and knobs

How a run is put together, in every domain here. The terms name what a thing *is* to the code that
uses it, never how it is built.

**Math object**: An injected collaborator that stands for a named concept and carries behaviour: a
loss, a projector, a penalty, a step rule, a recorder, a problem. Never a bag of numbers. _Avoid_:
component, helper, strategy

**Knob**: One scalar or name setting, owned by the object it parameterises: a learning rate belongs
to the step rule, a ridge epsilon to the projector, a step budget to the optimizer. _Avoid_:
hyperparameter (kept for the grid's own vocabulary), option, flag

**Configuration**: The knobs that produced a measurement: the algorithm's own and those of every
math object injected into it. A row recorded without its configuration cannot be reproduced.
_Avoid_: settings, params

**Composition root**: The script or harness that instantiates concrete classes and wires them
together. The only place a concrete class is named by name. _Avoid_: driver, runner, main

### Optimization

The descent side of the library, as the search side has its own vocabulary above. A run of an
optimizer is measured the way a search is: by what it paid, how it was set up and what it delivered.

**Optimizer**: The algorithm that runs a descent loop over a set of parameters, the analogue of a
search. Never the update rule it applies. _Avoid_: trainer, solver, Adam (a step rule)

**Step rule**: The update an optimizer applies at each step, with its schedule: Adam with a cosine
schedule is one step rule. Injected into the optimizer. _Avoid_: optimizer (taken: the loop),
scheduler alone

**Stop reason**:
Why a run ended: its step budget, a non-finite value, or a constraint a step violated. A run that
stops early is still a run and delivers a result carrying its reason; it never raises for a
numeric stop.
_Avoid_: error, failure, exception

**Training loss**: The scalar an optimizer descends: a cost plus its penalties. A training knob may
enter it; a reported number never comes from it. _Avoid_: objective (taken: the search's word, and
the relaxed objective is a different quantity)

### Recording and walkthroughs

**Recorder**:
An injected collaborator an algorithm hands its state to at the moments its state advances; off by
default (`NullRecorder`), never part of a result.
_Avoid_: state recorder, observer, logger, hook

**Frame**:
One recorded moment of a run — for a search one expansion, for an optimizer one step — with a caption
a reader can follow.
_Avoid_: snapshot, state (taken: a search configuration), trace (taken: residual trace)

**Recording**:
The frames of one run plus its metadata and the run's own result, as a versioned JSON document written
when the run ends.
_Avoid_: history, log

**Walkthrough**:
A recording rendered as a single offline HTML page with a stepper.
_Avoid_: dashboard, animation

**Key moment**:
A frame a view derives from the recording as one that decides the run, marked on the stepper.
_Avoid_: highlight, milestone, bookmark

**Narration**:
The sentence a view derives for a frame from that frame and the one before it; never authored per
run.
_Avoid_: caption (the recorder's own sentence, a different thing), commentary
