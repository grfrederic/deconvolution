Mathematical description of color deconvolution
===============================================

Introduction
------------

Let us imagine a light source that sends light through layers of different substances - each of them absorbing some of the light passing through. 
Given a digital recording of the light that passed through this setup (and information of the input light), we would like to reconstruct the layout of the substances.

Each pixel is represented by a three-component vector (it's RGB components). I will make frequent use of the Hadamard (or entry-wise) product:

.. math::
  \begin{bmatrix}
    x_1\\
    x_2\\
    x_3
  \end{bmatrix}
  \bullet
  \begin{bmatrix}
    y_1\\
    y_2\\
    y_3
  \end{bmatrix}=
  \begin{bmatrix}
    x_1y_1\\
    x_2y_2\\
    x_3y_3
  \end{bmatrix}

The Hadamard product is similar to the standard multiplication of real numbers and I will use the same notation for inverses and powers.

Consider light represented by a RGB-vector :math:`\vec i` passing through a unit-wide layer of some substance. I will assume that Beer's law [LB]_ hold's, 
that is the RGB-vector of the outgoing light will be:

.. math::
  \vec i \bullet \vec s=
  \begin{bmatrix}
    i_1s_1\\
    i_2s_2\\
    i_3s_3
  \end{bmatrix},

where :math:`\vec s` is specific for this substance (the choice of "unitary width" **does** change this value; this issue will be addressed later on). 
If, for example, :math:`\vec s=(1,1,1)` the layer does not absorb any light, for :math:`\vec s=(0,1,1)` the red channel is completely absorbed, but the remaining channels are untouched. 
Should the layer be :math:`a` times wider, the RGB components of the outgoing light would be:

.. math::
    \vec i \bullet \vec s^{\ a}=\begin{bmatrix}
    i_1\cdot s_1^a\\
    i_2\cdot s_2^a\\
    i_3\cdot s_3^a
    \end{bmatrix},

for multiple substances,

.. math::
	\vec i \bullet \vec p^{\ a} \bullet \vec q^{\ b} \bullet ...=\begin{bmatrix}
    i_1\cdot p_1^a \cdot q_1^b \cdot ...\\
    i_2\cdot p_2^a \cdot q_2^b \cdot ...\\
    i_3\cdot p_3^a \cdot q_3^b \cdot ...
    \end{bmatrix}.

This equation implies that changing the order of layers or splitting some of them does not change the outgoing light. I will work under the assumption that this also holds for mixed substances.

Ruifrok-Johnston deconvolution
------------------------------
In the case studied by Ruifrok and Johnston [RJ]_, the light :math:`\vec i` passes through three substances. Vectors :math:`($\vec v$, $ \vec u$, $\vec w$)` describing absorption rates for all substances 
(that is, absorption coefficients for unit-wide substance layers) are assumed to be known. The width of each substance layer (which may change from point to point) 
has to be calculated given the output light. Suppose the camera registers a single vector :math:`\vec r` at some arbitrary pixel. We wish to express this vector according to the equation:

.. math::
  \label{eq:decon3}
  \vec i \bullet \vec v^{\ a} \bullet \vec u^{\ b} \bullet \vec w^{\ c} = \vec r.

Solving this equation for :math:`a, b` and :math:`c` gives the layers widths in unit lengths. We may then compute how much light would have passed through each layer separately: 
the first deconvolution is just :math:`\vec i \bullet \vec v^{\ a}`, the second :math:`\vec i \bullet \vec u^{\ b}` and the third :math:`\vec i \bullet \vec w^{\ c}`. 
It is possible that no real non-negative :math:`a, b, c` solve this equation (due to data noise, imperfect digitization, traces of other substances, etc.). 
In that case, the reconstructed image will differ from the original: this difference can be visualized by considering "rest picture":

.. math::
  \vec r \bullet \vec v^{\ -a} \bullet \vec u^{\ -b} \bullet \vec w^{\ -c}.

Let us briefly return to the issue of picking a unit width. Notice that changing the unit width of a substance by a factor of :math:`\lambda`, changes the constant $a$ at each pixel to 
:math:`a/ \lambda`, and thus the density distribution :math:`a` times the unit width does not change. Similarly, the deconvolved images don't change:

.. math::
  \vec i \bullet \vec v^{\ a} = \vec i \bullet (\vec v^{\ \lambda})^{a/ \lambda}.

Equation (\ref{eq:decon3}) is in fact a set of three equations - one for each component. Lets rewrite it for the :math:`k`-th component and transform equivalently:

.. math::
  v_k^a\cdot u_k^b \cdot w_k^c &= \frac{r_k}{i_k}\\
  a \log v_k + b \log u_k  + c \log w_k &= \log(r_k/i_k),

going back to vector representation:

.. math::
  \label{eq:3vec}
  \begin{bmatrix}
    \log v_1 & \log u_1 & \log w_1\\
    \log v_2 & \log u_2 & \log w_2\\
    \log v_3 & \log u_3 & \log w_3
  \end{bmatrix}
  \begin{bmatrix}
    a\\
    b\\
    c
  \end{bmatrix}
  -
  \begin{bmatrix}
    \log(r_1/i_1)\\\log(r_2/i_2)\\\log(r_3/i_3)
  \end{bmatrix}=0.

This equation is solvable if and only if the left matrix is invertible, and :math:`i_k \neq 0`. In physical terms, the first assumption states that no substance can be "faked" by mixing the remaining two, 
and the second that the input light is nonzero for all channels.

Given an input image, at each pixel :math:`\vec r` we may solve for :math:`a,b,c`. Should we get any negative results, it means that this particular pixels color can not be obtained by mixing
the given substances. In this case the assumption that the image was obtained by mixing the given substances is violated - hence it's reasonable to disregard such results data noise and drop
all negative parts. Having that done, we are able to construct five pixels:

- Reconstructed pixel: :math:`\vec i \bullet \vec v^{\ a} \bullet \vec u^{\ b} \bullet \vec w^{\ c}`
- Difference from the original image, due to negative cut-off: :math:`\vec r \bullet \vec v^{\ -a} \bullet \vec u^{\ -b} \bullet \vec w^{\ -c}`
- Three single-substance pixels:
  + :math:`\vec i \bullet \vec v^{\ a}`
  + :math:`\vec i \bullet \vec u^{\ b}`
  + :math:`\vec i \bullet \vec w^{\ c}`

After processing every pixel in this manner, the reconstructed image, three single substance images, and one remainder image (showing the error) are obtained. *An example is shown in Figure \ref{fig:ruifrok}.*

Until now the approach was based on Ruifrok and Johnston - however, the choice of formalism makes it easier to look for further development. 
Handle any number of channels straightforward: in fact, for the general case, the notation does not change. 
Secondly, if only two substances are of interest, Ruifrok and Johnston suggest measuring the absorbances of those two substances, and then choosing the third so that it minimizes the negative cut-off. 
The third single-substance image is then used as a measure of error. This arbitrarity seems a bit artificial - now I introduce a method developed by Frederic Grabowski.

Two-substance deconvolution
---------------------------

Choosing the third substance by hand so that it minimizes data loss due to negative cut-offs introduces ambiguity into the measurement, and seems artificial. 
In order to fix this problem, drop the third substance entirely, and look for :math:`a, b` that minimize the squared error of the approximation:

.. math::
  \begin{bmatrix}
    \log v_1 & \log u_1 \\
    \log v_2 & \log u_2 \\
    \log v_3 & \log u_3 
  \end{bmatrix}
  \begin{bmatrix}
    a\\
    b
  \end{bmatrix}
  -
  \begin{bmatrix}
    \log(r_1/i_1)\\\log(r_2/i_2)\\\log(r_3/i_3)
  \end{bmatrix} \approx 0.
  :label: 2vec

Where both the matrix and the 3-vector are given. Clean up the notation: 

.. math::
  \inf_{\vec x\ \in\ \mathbb{R}^2} f(x) = \inf_{\vec x\ \in\ \mathbb{R}^2}||A\vec x-\vec y||^2

We want to know the :math:`\vec x` for which this infimum is obtained. This :math:`\vec x` always exists, because it is the orthogonal projection of :math:`\vec y` onto :math:`A(\mathbb{R}^2)`.

For each minimum :math:`\text{grad}\, f=0`. That is, for the first component:

.. math::
  0=\frac12 \frac{\partial f}{\partial x_1} (x_1,x_2)=\sum_{k=1}^3 A_{k1}\left(A_{k1}x_1+A_{k2}x_2-r_k\right) \\
  x_1\sum_{k=1}^3 A_{k1}^2 + x_2\sum_{k=1}^3 A_{k1}A_{k2} = \sum_{k=1}^3 A_{k1}r_k
  :label: solAB

Combining (\ref{eq:solAB}) with the equation for the second component we get a set of two linear equations:

.. math::
  \sum_k
  \begin{bmatrix}
    A_{k1}^2 & A_{k1}A_{k2} \\
    A_{k1}A_{k2} & A_{k2}^2
  \end{bmatrix}
  \begin{bmatrix}
    x_1\\
    x_2
  \end{bmatrix}
  =
  \begin{bmatrix}
    \sum_k A_{k1}r_k\\
    \sum_k A_{k2}r_k
  \end{bmatrix}.

The equation above can be easily solved if and only if it's determinant is not 0:

.. math::
  \det \begin{bmatrix}
    \sum_k A_{k1}^2 & \sum_k A_{k1}A_{k2} \\
    \sum_k A_{k1}A_{k2} & \sum_k A_{k2}^2
  \end{bmatrix} = \\
  \sum_k A_{k1}^2 \cdot \sum_k A_{k2}^2 - \left(\sum_k A_{k1}A_{k2} \right)^2 \neq 0

The Cauchy-Schwarz inequality states that the considered determinant is 0 if and only if there is a number :math:`t` for which :math:`A_{k1}=t\cdot A_{k2}`. 
This is again the mixing independence of the basis. If :math:`\dim A(\mathbb{R}^2) <2`, then there does not exist a unique :math:`\vec x` for which the projected vector is obtained.

We now have a method for finding "the best" :math:`a,\ b` solving equation (\ref{eq:2vec}). This means that for each pixel and a basis of two given substances, we are able to calculate four pixels: 
- Best reconstructed pixel: :math:`\vec i \bullet \vec v^{\ a} \bullet \vec u^{\ b}`
- Difference from the original image: :math:`\vec r \bullet \vec v^{\ -a} \bullet \vec u^{\ -b}`
- Two single-substance pixels: :math:`\vec i \bullet \vec v^{\ a}` and :math:`\vec i \bullet \vec u^{\ b}` 

After processing every pixel in this manner the reconstructed image, two single substance images, and one remainder image (showing the error) are obtained. *An example is shown in Figure \ref{fig:auto}.*

Formulation of the optimization problem
---------------------------------------

Considering deconvolutions with two substances has another advantage - it gives a criterium for comparing bases. Taking 

.. math::
  \sum_{p\ \in\ pixels}\  \inf_{a,b,c\ \in\ \mathbb{R}} ||\text{LHS\ of eq.}(\ref{eq:3vec})||^2

does not work, because the equation is always soluble and the expression is identically zero (at least for all independent bases). 
Decreasing the number of degrees of freedom (that is, the number of substances to match) solves this difficulty:

.. math::
  \begin{bmatrix}
    \log v_1 & \log u_1 \\
    \log v_2 & \log u_2 \\
    \log v_3 & \log u_3 
  \end{bmatrix}
  \begin{bmatrix}
    a\\
    b
  \end{bmatrix}
  -
  \begin{bmatrix}
    \log(r_1/i_1)\\\log(r_2/i_2)\\\log(r_3/i_3)
  \end{bmatrix} \approx 0.

.. math::
  \sum_{p\ \in\ pixels}\  \inf_{a,b\ \in\ \mathbb{R}} \lvert\lvert  
  \begin{bmatrix}
    \log v_1 & \log u_1 \\
    \log v_2 & \log u_2 \\
    \log v_3 & \log u_3 
  \end{bmatrix}
  \begin{bmatrix}
    a\\
    b
  \end{bmatrix}
  -
  \begin{bmatrix}
    \log(r_1/i_1)\\\log(r_2/i_2)\\\log(r_3/i_3)
  \end{bmatrix} \rvert\rvert^2
  :label: criterium

To solve this optimization problem, first clean up the notation. Let:

.. math::
  A = 
  \begin{bmatrix}
    a_{11} & a_{12} \\
    a_{21} & a_{22} \\
    a_{31} & a_{32}
  \end{bmatrix},

.. math::
  \begin{bmatrix}
    x_1\\
    x_2
  \end{bmatrix}= \vec x = 
  \begin{bmatrix}
    a\\
    b
  \end{bmatrix}

.. math::
  \begin{bmatrix}
    y_1(p)\\
    y_2(p)\\
    y_3(p)
  \end{bmatrix}= \vec y(p) = 
  \begin{bmatrix}
    \log(r_1/i_1)\\\log(r_2/i_2)\\\log(r_3/i_3)
  \end{bmatrix}

Given some :math:`\vec y(p)`, the problem is to find a :math:`2\times3` matrix :math:`A`, that minimizes the expression:

.. math::
  \label{eq:optOrig}
  f(A) = \sum_{p\ \in\ pixels}\  \inf_{\vec x\ \in\ \mathbb{R}^2}||A\vec x-\vec y(p)||^2

Solving the optimization problem
--------------------------------

For any :math:`A` in equation \ref{eq:optOrig}:

.. math::
  \inf_{\vec x\ \in\ \mathbb{R}^2}||A\vec x-\vec y(p)||^2 = d(\vec y(p),\ A(\mathbb{R}^2))^2,

hence we want to minimize the mean squared distance of the points $y$ from the image space of :math:`A`. 
There are two cases: either :math:`\dim A(\mathbb{R}^2) = 2` or is strictly less than 2. In the second case, we are always able to choose such a matrix :math:`A`, 
that the previous image is a subspace of the new image, but then the distances can only be smaller. 
It thus suffices to find the best two dimensional space. Every such space has a normal vector, which we choose so that the third component in non-negative 
(this convention is arbitrary, but does not matter). 
We can now rewrite (\ref{eq:optOrig}) as:

.. math::
  \label{eq:optG}
  f(A) = g(\vec n) = \sum_{p\ \in\ pixels}\  \lvert\vec n \cdot \vec y(p)\rvert^2

Any :math:`\vec n` minimizing (\ref{eq:optG}), determines a class of matrices minimizing (\ref{eq:optOrig}) - precisely those, whose image is perpendicular to :math:`\vec n`.
We only need to consider :math:`\vec n` such that :math:`||\vec n||=1`. Because the set of such :math:`\vec n` is compact in :math:`\mathbb{R}^3`, we can apply the method of Lagrange multipliers:

.. math::
  \nabla ( g(\vec n) -\lambda ||\vec n||^2 ) = 0,
after expanding and rearranging

.. math::
  \label{eq:eigenOrig}
  \sum_{p\ \in\ pixels}\ (\ \vec y(p) \cdot \vec n\ )\ \vec y(p) = \lambda \vec n.

The left hand side is a linear operator from :math:`\mathbb{R}^3` to :math:`\mathbb{R}^3` applied to :math:`\vec n`. Equation *(\ref{eq:eigenOrig})* is just the eigenvalue equation for this operator. 
Moreover, multiplying both sides by :math:`\vec n` we notice :math:`\lambda = g(\vec n)`. The smallest-eigenvalue eigenvector is the 
:math:`\vec n` minimizing :math:`g(\vec n)`, and the corresponding eigenvalue the value :math:`g(\vec n)`.

Computing the deconvolution
---------------------------

Rewrite the left hand side of equation *(\ref{eq:eigenOrig})* using index notation:

.. math::
  \sum_{p\ \in\ pixels}\ \sum_{i=1}^3 y_i(p) y_j(p)\cdot n_i,

is the :math:`j`-th component of the resulting vector. Hence the matrix of the linear transformation is:

.. math::
 (Y)_{ij} = \sum_{p\ \in\ pixels}\ \sum_{i=1}^3 y_i(p) y_j(p)

Given an input image, first calculate :math:`Y`, and find it's eigenvalue decomposition. Pick the eigenvector :math:`\vec n` with smallest eigenvalue. 
Choosing an :math:`A` such that :math:`\vec n` is perpendicular to :math:`A`'s image is equivalent to choosing a basis with both elements perpendicular to :math:`\vec n`. 
To have any preference, let's return to the physical interpretation. Naively, all these bases allow us to mix the same set of colors:
but not for all bases will this mixing will be physically meaningful. Consider the following example: the basis consisting of two substances, one absorbing only red the other only blue, 
will be equivalent to a basis of one substance absorbing both red and blue, and the other only blue. However, only in the first base is it possible to construct a color with only the red 
channel absorbed. This happens because we cannot physically have negative widths of substances. It seems advantageous to choose the basis that allows us to mix the widest range of colors physically. 
It turns out that this choice is not always optimal. For now, we stick with the maximal physical color range. 
The basis of our choosing is the one for which both vectors are non-negative (so that the resulting substances absorb light, and not amplify it), and have the biggest angle between them. 
This determines them uniquely up to a rearrangement.


References
----------

.. [LB] Modern analysis of this law can be found in `"Employing Theories Far beyond Their Limits—The Case of the (Boguer-) Beer–Lambert Law" by Mayerhoefer et al.
  <http://onlinelibrary.wiley.com/doi/10.1002/cphc.201600114/abstract>`_
.. [RJ] `Research paper by Ruifrok and Johnston
  <https://www.researchgate.net/publication/11815294_Ruifrok_AC_Johnston_DA_Quantification_of_histochemical_staining_by_color_deconvolution_Anal_Quant_Cytol_Histol_23_291-299>`_
