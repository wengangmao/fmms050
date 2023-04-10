\section*{Project 1: Roll decay damping}
\label{project1}

A parameters identification technique (PIT) is often used to obtain the damping coefficients from the roll decay tests. In this technique, parameters in a mathematical
model are determined in order to get the best fit to a roll decay time
signal. A derivation of a mathematical model suitable for this study is
described below together with a description of how the parameters:
the inertia coefficients $A_{44}$, the damping $B_{44}$ and the stiffness $C_{44}$ as in Eq.\ref{eq:roll_decay_equation_general_himeno} are determined. The roll
decay motion can be expressed in general form  but with nonlinear stiffness:
\begin{equation}
A_{44} \ddot{\phi} + \operatorname{B_{44}}\left(\dot{\phi}\right) + \operatorname{C_{44}}\left(\phi\right) = 0,
\label{eq:roll_decay_equation_general_himeno}
\end{equation}
where $B_{44}(\dot{\phi})$ and $C_{44}(\phi)$ are the damping and
stiffness models. A cubic model can be obtained by using cubic damping:
\begin{equation}
\operatorname{B_{44}}\left(\dot{\phi}\right) = B_{1} \dot{\phi} + B_{2} \left|{\dot{\phi}}\right| \dot{\phi} + B_{3} \dot{\phi}^{3}
\label{eq:b44_cubic_equation}
\end{equation}
And a higher order stiffness model:
\begin{equation}
\operatorname{C_{44}}\left(\phi\right) = C_{1} \phi + C_{3} \phi^{3} + C_{5} \phi^{5}
\label{eq:restoring_equation_cubic}
\end{equation}
The total equation is then written:
\begin{equation}
A_{44} \ddot{\phi} + \left(B_{1} + B_{2} \left|{\dot{\phi}}\right| + B_{3} \dot{\phi}^{2}\right) \dot{\phi} + \left(C_{1} + C_{3} \phi^{2} + C_{5} \phi^{4}\right) \phi = 0
\label{eq:roll_decay_equation_cubic}
\end{equation}
NB: this equation does not have one unique solution however. If all parameters would be multiplied
by a factor $k$ these parameters would also yield as a solution to the equation. 

The parameters of this equation can be identified using least square fit if the time signals $\phi(t)$, $\dot{\phi}(t)$ and
$\ddot{\phi}(t)$ are all known. For model tests, where only the roll signal $\phi(t)$ is known, the other time derivatives can be estimated
using numerical differentiation of a low-pass filtered roll signal or
Kalman filtered roll signal. The filtering will however introduce some
errors in itself. So instead of using this ``differentiation approach",
it has been found that solving the differential equation numerically for
estimated parameter values determined using optimization. One problem with this ``Integration
approach" is that in order to converge, the optimization needs a
reasonable first guess of the parameters. The Differentiation approach
has therefore been used as a pre-step to obtain a very good first guess
of the parameters that can be passed on to the Integration approach.\\

\begin{comment}
The differential equation is numerically solved as an initial
value problem, where the initial states for $\phi(t)$,
$\dot{\phi}(t)$ and $\ddot{\phi}(t)$ are used to estimate the
following states, by conducting very small time steps using the
following expression for the acceleration:
\begin{equation}
\ddot{\phi} = - B_{1A} \dot{\phi} - B_{2A} \left|{\dot{\phi}}\right| \dot{\phi} - B_{3A} \dot{\phi}^{3} - C_{1A} \phi - C_{3A} \phi^{3} - C_{5A} \phi^{5}
\label{eq:eq_phi1d}
\end{equation}
This numerical solution can be compared with an analytical solution for
a linear model.  For this case the relation between $\zeta$ and $B_1$ can
be expressed as:
\begin{equation}
B_{1} = 2 A_{44} \omega_{0} \zeta
\label{eq:B_1_zeta_eq}
\end{equation}
and the natural frequency can be obtained from:
\begin{equation}
\omega_{0} = \sqrt{\frac{C_{1}}{A_{44}}}
\label{eq:omega0_eq}
\end{equation}
The analytical and numerical solutions are very similar according to the
example: $A_{44} = 1.0$, $B_1 = 0.3$, $C_1 = 5.0$ shown in Fig.\ref{fig:analytical_numerical}.
\begin{figure}[H]
\begin{center}\includegraphics[width = 0.5\textwidth]{figures/analytical_numerical.pdf}\end{center}
\vspace{-1cm}
\caption{Comparison of analytical solution and numerical simulation of a roll decay test.}
\label{fig:analytical_numerical}
\end{figure}
\end{comment}


\subsection*{\textbf{Roll damping estimation for KVLCC}}

Your task is to use different statistical regression methods to obtain different parameters in the Roll decay Eq.(\ref{eq:roll_decay_equation_cubic})
, especially how to get the roll damping, and discuss how sensitive of the estimation of the roll damping.

\subsection*{Test at 0 knots}\label{test-at-0-knots}
Two roll decay model tests were conducted at zero speed referred to as Run 1 and 2.
These tests where analyzed by fitting a cubic model
to the model test data. The two models were very similar in terms of roll damping and stiffness, suggesting good repeatability in both the model tests and in the parameter
identification technique (PIT) used. It can be seen that the dampings,
from each individual oscillation obtained with the logarithmic decrement
method, are very scattered. This scatter does not seem to influence the
two models for the 0 speed case, which are very similar.
\subsubsection*{Test at 15.5 knots}\label{test-at-15.5-knots}
One roll decay model tests, referred to as Run 3, was conducted at a ship speed corresponding to
15.5 knots full scale ship speed. The
ship got a small yaw rate
at the end of test, giving a small steady roll angle due to the
centrifugal force. Since this effect is not included in the mathematical
model used, the steady roll angle was instead removed by removing the
linear trend in the roll angle signal.


{\color{blue}NB:You can download data from ``https://data.mendeley.com/datasets/2stvkyngj9/2", or ``https://chalmersuniversity.box.com/s/wsnew9tz6bj9yzbs86i967p6l93im6zm".
}