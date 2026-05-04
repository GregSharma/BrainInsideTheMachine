Problem 1
Andy and Betsy both live in Mathville. Andy leaves Mathville on his bicycle at $1{:}30$, traveling due north at a steady $8$ miles per hour. Betsy leaves on her bicycle from the same point at $2{:}30$, traveling due east at a steady $12$ miles per hour. At what time will they be exactly the same distance from their common starting point?

$\textbf{(A) } 3{:}30 \qquad\textbf{(B) } 3{:}45 \qquad\textbf{(C) } 4{:}00 \qquad\textbf{(D) } 4{:}15 \qquad\textbf{(E) } 4{:}30$

Solution

Problem 2
A box contains $10$ pounds of a nut mix that is $50$ percent peanuts, $20$ percent cashews, and $30$ percent almonds. A second nut mix containing $20$ percent peanuts, $40$ percent cashews, and $40$ percent almonds is added to the box resulting in a new nut mix that is $40$ percent peanuts. How many pounds of cashews are now in the box?

$\textbf{(A)}~3.5\qquad\textbf{(B)}~4\qquad\textbf{(C)}~4.5\qquad\textbf{(D)}~5\qquad\textbf{(E)}~6$

Solution

Problem 3
A team of students is going to compete against a team of teachers in a trivia contest. The total number of students and teachers is $15$. Ash, a cousin of one of the students, wants to join the contest. If Ash plays with the students, the average age on that team will increase from $12$ to $14$. If Ash plays with the teachers, the average age on that team will decrease from $55$ to $52$. How old is Ash?

$\textbf{(A)}~28\qquad\textbf{(B)}~29\qquad\textbf{(C)}~30\qquad\textbf{(D)}~32\qquad\textbf{(E)}~33$

Solution

Problem 4
Agnes writes the following four statements on a blank piece of paper.

At least one of these statements is true.
At least two of these statements are true.
At least two of these statements are false.
At least one of these statements is false.
Each statement is either true or false. How many false statements did Agnes write on the paper?

$\textbf{(A)}~0\qquad\textbf{(B)}~1\qquad\textbf{(C)}~2\qquad\textbf{(D)}~3\qquad\textbf{(E)}~4$

Solution

Problem 5
In the figure below, the outside square contains infinitely many squares, each of them with the same center and sides parallel to the outside square. The ratio of the side length of a square to the side length of the next inner square is $k$, where $0 < k < 1.$ The spaces between squares are alternately shaded as shown in the figure (which is not necessarily drawn to scale).[asy] unitsize(1cm);  int n = 25;               real s = 6;               real ratio = 0.767;  //definitely not on purpose      real a = s;                for (int i = 0; i < n; ++i) {   real b = a * ratio;                         // Draw current square   draw(box((-a/2,-a/2),(a/2,a/2)));     if (i % 2 == 1) {     fill(box((-a/2,-a/2),(a/2,a/2)), gray(1));      } else {     fill(box((-a/2,-a/2),(a/2,a/2)), lightred);      }      a = b;    }   draw(box((-a/2,-a/2),(a/2,a/2))); [/asy]

The area of the shaded portion of the figure is $64\%$ of the area of the original square. What is $k$?

$\textbf{(A) } \frac{3}{5} \qquad\textbf{(B) } \frac{16}{25} \qquad\textbf{(C) } \frac{2}{3} \qquad\textbf{(D) } \frac{3}{4} \qquad\textbf{(E) } \frac{4}{5}$

Solution

Problem 6
Six chairs are arranged around a round table. Two students and two teachers randomly select four of the chairs to sit in. What is the probability that the two students will sit in two adjacent chairs and the two teachers will also sit in two adjacent chairs?

$\textbf{(A) } \frac 16 \qquad \textbf{(B) } \frac 15 \qquad \textbf{(C) } \frac 29 \qquad \textbf{(D) } \frac 3{13} \qquad \textbf{(E) } \frac 14$

Solution

Problem 7
In a certain alien world, the maximum running speed $v$ of an organism is dependent on its number of toes $n$ and number of eyes $m$. The relationship can be expressed as $v=kn^am^b$ centimeters per hour, where $k$, $a$, and $b$ are integer constants. In a population where all organisms have $5$ toes, $\log v=4+2\log m$; and in a population where all organisms have $25$ eyes, $\log v=4+4\log n$, where the logarithms are base 10. What is $k+a+b$?

$\textbf{(A)}~20\qquad\textbf{(B)}~21\qquad\textbf{(C)}~22\qquad\textbf{(D)}~23\qquad\textbf{(E)}~24$

Solution

Problem 8
Pentagon $ABCDE$ is inscribed in a circle, and $\angle BEC = \angle CED = 30^\circ$. Let line $AC$ and line $BD$ intersect at point $F$, and suppose that $AB = 9$ and $AD = 24$. What is $BF$?

$\textbf{(A) } \frac{57}{11} \qquad\textbf{(B) } \frac{59}{11} \qquad\textbf{(C) } \frac{60}{11} \qquad\textbf{(D) } \frac{61}{11} \qquad\textbf{(E) } \frac{63}{11}$

Solution

Problem 9
Let $w$ be the complex number $2+i$, where $i=\sqrt{-1}$. What real number $r$ has the property that $r$, $w$, and $w^2$ are three collinear points in the complex plane?

$\textbf{(A)}~\frac34\qquad\textbf{(B)}~1\qquad\textbf{(C)}~\frac75\qquad\textbf{(D)}~\frac32\qquad\textbf{(E)}~\frac53$

Solution

Problem 10
In the figure shown below, major arc $\widehat{AD}$ and minor arc $\widehat{BC}$ have the same center, $O$. Also, $A$ lies between $O$ and $B$, and $D$ lies between $O$ and $C$. Major arc $\widehat{AD}$, minor arc $\widehat{BC}$, and each of the two segments $\overline{AB}$ and $\overline{CD}$ have length $2\pi$. What is the distance from $O$ to $A$?

[asy] usepackage("mathptmx"); size(6cm); defaultpen(linewidth(0.7)); real r = 1 - pi + sqrt(pi^2 + 1); pair O = (0, 0), A = r * dir(-30), B = (r + 2pi) * dir(-30), C = (r + 2pi) * dir(30), D = r * dir(30); draw(arc(O, r, 30, 330)); draw(arc(O, r + 2pi, -30, 30)); draw(O--A, dashed); draw(O--D, dashed); draw(A--B); draw(C--D); dot("$O$", O, dir(135)); dot("$A$", A, dir(270)); dot("$B$", B, dir(0)); dot("$C$", C, dir(0)); dot("$D$", D, 1.5 * dir(70)); [/asy]

$\textbf{(A)}~1 \qquad \textbf{(B)}~1 - \pi + \sqrt{\pi^{2} + 1} \qquad \textbf{(C)}~\frac{\pi}{2} \qquad \textbf{(D)}~\frac{\sqrt{\pi^{2} + 1}}{2} \qquad \textbf{(E)}~2$

Solution

Problem 11
The orthocenter of a triangle is the concurrent intersection of the three (possibly extended) altitudes. What is the sum of the coordinates of the orthocenter of the triangle whose vertices are $A(2,31), B(8,27),$ and $C(18,27)$?

$\textbf{(A)}~5\qquad\textbf{(B)}~17\qquad\textbf{(C)}~10+4\sqrt{17} +2\sqrt{13}\qquad\textbf{(D)}~\frac{113}{3}\qquad\textbf{(E)}~54$

Solution

Problem 12
The harmonic mean of a collection of numbers is the reciprocal of the arithmetic mean of the reciprocals of the numbers in the collection. For example, the harmonic mean of $4,4,$ and $5$ is

\[\frac{1}{\frac{1}{3}(\frac{1}{4}+\frac{1}{4}+\frac{1}{5})}=\frac{30}{7}.\]

What is the harmonic mean of all the real roots of the $4050$th degree polynomial

\[\prod_{k=1}^{2025} (kx^2-4x-3)=(x^2-4x-3)(2x^2-4x-3)(3x^2-4x-3)...(2025x^2-4x-3)?\]

$\textbf{(A)}~-\frac{5}{3}\qquad\textbf{(B)}~-\frac{3}{2}\qquad\textbf{(C)}~-\frac{6}{5}\qquad\textbf{(D)}~-\frac{5}{6}\qquad\textbf{(E)}~-\frac{2}{3}$

Solution

Problem 13
Let $C = \{1, 2, 3, \dots, 13\}$. Let $N$ be the greatest integer such that there exists a subset of $C$ with $N$ elements that does not contain five consecutive integers. Suppose $N$ integers are chosen at random from $C$ without replacement. What is the probability that the chosen elements do not include five consecutive integers?

$\textbf{(A)}~\frac{3}{130} \qquad \textbf{(B)}~\frac{3}{143} \qquad \textbf{(C)}~\frac{5}{143} \qquad \textbf{(D)}~\frac{1}{26} \qquad \textbf{(E)}~\frac{5}{78}$

Solution

Problem 14
Points $F$, $G$, and $H$ are collinear with $G$ between $F$ and $H$. The ellipse with foci at $G$ and $H$ is internally tangent to the ellipse with foci at $F$ and $G$, as shown below.

[asy] /* Made by Mathemagician108 */ import graph; unitsize(0.15 inch); draw(ellipse((0,0), 12, 4)); draw(ellipse((8,0), 4, 1.45)); pair F = (-6,0); pair G = (6,0); pair H = (10,0); dot(F^^G^^H); label("$F$", F, 1.5*S, p=fontsize(8pt));  label("$G$", G, 1.5*S, p=fontsize(8pt));  label("$H$", H, 1.5*S, p=fontsize(8pt)); [/asy]

The two ellipses have the same eccentricity $e$, and the ratio of their areas is $2025$. (Recall that the eccentricity of an ellipse is $e = \tfrac{c}{a}$, where $c$ is the distance from the center to a focus, and $2a$ is the length of the major axis.) What is $e$?

$\textbf{(A)}~\frac35\qquad\textbf{(B)}~\frac{16}{25}\qquad\textbf{(C)}~\frac45\qquad\textbf{(D)}~\frac{22}{23}\qquad\textbf{(E)}~\frac{44}{45}$

Solution

Problem 15
A set of numbers is called sum-free if whenever $x$ and $y$ are (not necessarily distinct) elements of the set, $x+y$ is not an element of the set. For example, $\{1,4,6\}$ and the empty set are sum-free, but $\{2,4,5\}$ is not. What is the greatest possible number of elements in a sum-free subset of $\{1,2,3,...,20\}$?

$\textbf{(A) } 8 \qquad\textbf{(B) } 9 \qquad\textbf{(C) } 10 \qquad\textbf{(D) } 11 \qquad\textbf{(E) } 12$

Solution

Problem 16
Triangle $\triangle ABC$ has side lengths $AB = 80$, $BC = 45$, and $AC = 75$. The bisector $\angle B$ and the altitude to side $\overline{AB}$ intersect at point $P$. What is $BP$?

$\textbf{(A)}~18\qquad\textbf{(B)}~19\qquad\textbf{(C)}~20\qquad\textbf{(D)}~21\qquad\textbf{(E)}~22$

Solution

Problem 17
The polynomial $(z + i)(z + 2i)(z + 3i) + 10$ has three roots in the complex plane, where $i = \sqrt{-1}$. What is the area of the triangle formed by these three roots?

$\textbf{(A)}~6 \qquad \textbf{(B)}~8 \qquad \textbf{(C)}~10 \qquad \textbf{(D)}~12 \qquad \textbf{(E)}~14$

Solution

Problem 18
How many ordered triples $(x, y, z)$ of different positive integers less than or equal to $8$ satisfy $xy > z$, $xz > y$, and $yz > x$?

$\textbf{(A)}~36 \qquad \textbf{(B)}~84 \qquad \textbf{(C)}~186 \qquad \textbf{(D)}~336 \qquad \textbf{(E)}~486$

Solution

Problem 19
Let $a$, $b$, and $c$ be the roots of the polynomial $x^3 + kx + 1$. What is the sum\[a^3b^2 + a^2b^3 + b^3c^2 + b^2c^3 + c^3a^2 + c^2a^3?\]

$\textbf{(A)}~-k\qquad\textbf{(B)}~-k+1\qquad\textbf{(C)}~1\qquad\textbf{(D)}~k-1\qquad\textbf{(E)}~k$

Solution

Problem 20
The base of the pentahedron shown below is a $13 \times 8$ rectangle, and its lateral faces are two isosceles triangles with base of length $8$ and congruent sides of length $13$, and two isosceles trapezoids with bases of length $7$ and $13$ and nonparallel sides of length $13$.

[asy] /* Refined by Mathemagician108 */ import graph3; size(200); real l = 13; real w = 8; real offset = (l - 7)/2;  // 3 real midy = w/2;  // 4 real h = 12; triple O1 = (0,0,0); triple O2 = (l,0,0); triple O3 = (l,w,0); triple O4 = (0,w,0); triple T1 = (offset, midy, h); triple T2 = (l - offset, midy, h); currentprojection=orthographic((-4,-6,3)); draw(O4--O1--O2, linewidth(1)); draw(O2--O3--O4, dashed + linewidth(1)); draw(O3--T2, dashed + linewidth(1)); draw(O1--T1, linewidth(1)); draw(O4--T1, linewidth(1)); draw(O2--T2, linewidth(1)); draw(T1--T2, linewidth(1)); label("13", (O1+O2)/2, 3*-Y);  // Bottom length label("13", (O2+T2)/2, 1.5*X); label("13", (O4+T1)/2, 2*-X); label("8", (O1+O4)/2, 2*-X);   // Width label("7", (T1+T2)/2, 1.5*Z);    // Top length [/asy]

What is the volume of the pentahedron?

$\textbf{(A)}~416\qquad\textbf{(B)}~520\qquad\textbf{(C)}~528\qquad\textbf{(D)}~676\qquad\textbf{(E)}~832$

Solution

Problem 21
There is a unique ordered triple $(a,k,m)$ of nonnegative integers such that\[\frac{4^a + 4^{a+k}+4^{a+2k}+\cdots + 4^{a+mk}}{2^a + 2^{a+k} + 2^{a+2k}+ \cdots + 2^{a+mk}} = 964.\]What is $a+k+m$?

$\textbf{(A) } 8 \qquad \textbf{(B) } 9 \qquad \textbf{(C)}  10  \qquad  \textbf{(D) } 11 \qquad \textbf{(E) } 12$

Solution

Problem 22
Three real numbers are chosen independently and uniformly at random between $0$ and $1$. What is the probability that the greatest of these three numbers is greater than $2$ times each of the other two numbers? (In other words, if the chosen numbers are $a \geq b \geq c$, then $a > 2b$.)

$\textbf{(A)}~\frac{1}{12}\qquad\textbf{(B)}~\frac19\qquad\textbf{(C)}~\frac18\qquad\textbf{(D)}~\frac16\qquad\textbf{(E)}~\frac14$

Solution

Problem 23
Call a positive integer fair if no digit is used more than once, it has no $0$s, and no digit is adjacent to two greater digits. For example, $196, 23,$ and $12463$ are fair, but $1546, 320$ and $34321$ are not. How many fair positive integers are there?

$\textbf{(A) } 511 \qquad \textbf{(B) } 2584 \qquad \textbf{(C) } 9841 \qquad \textbf{(D) } 17711 \qquad \textbf{(E) } 19682$

Solution

Problem 24
A circle of radius $r$ is surrounded by $12$ circles of radius $1,$ externally tangent to the central circle and sequentially tangent to each other, as shown. Then $r$ can be written as $\sqrt a + \sqrt b + c,$ where $a, b, c$ are integers. What is $a+b+c?$

[asy] defaultpen(fontsize(12)+linewidth(1)); size(200); real r=2.925, x=360/12; pair O=origin; draw(CR(O,r),black+1.5); for (int i = 0; i<12; ++i) {  draw(CR((r+1)*dir(i*x),1)); } dot(O); dot((r+1)*right); draw(O--(r,0)^^(r+1,0)--(r+2,0), linewidth(0.5)); label("$r$",(r/2,0),up); label("$1$",(r+3/2,0),up); [/asy]

$\textbf{(A) } 3 \qquad \textbf{(B) } 5 \qquad \textbf{(C) } 7 \qquad \textbf{(D) } 9 \qquad \textbf{(E) } 11$

Solution

Problem 25
Polynomials $P(x)$ and $Q(x)$ each have degree $3$ and leading coefficient $1$, and their roots are all elements of $\{1,2,3,4,5\}$. The function $f(x) =\tfrac{P(x)}{Q(x)}$ has the property that there exist real numbers $a < b < c < d$ such that the set of all real numbers $x$ such that $f(x) \leq 0$ consists of the closed interval $[a,b]$ together with the open interval $(c,d)$. How many ordered pairs of polynomials $(P,Q)$ are possible? $\textbf{(A) } 9 \qquad \textbf{(B) } 11 \qquad \textbf{(C) } 12 \qquad \textbf{(D) } 13 \qquad \textbf{(E) } 8$

Note:
The original problem asked for the number of functions $f(x)$, which resulted the problem having an answer not present in the answer choices. I have added answer F to this problem, but it was not there in the actual contest.

To make sure there should be only 5, so I removed an answer choice A which... seemed reasonable to guessers

Solution

