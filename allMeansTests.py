import pandas as pd
from statistics import NormalDist
from statistics import variance
from scipy.stats import f
from scipy.stats import chi2
from numpy import log

def meansTest(data, groups, scores, test, alpha=0.05, iters=False, order=2, alt=False):
    '''
    meansTest
     
    This function can perform various one-way anovas (comparisons of means)
    
    Parameters
    ----------
    data : pandas dataframe
        A pandas dataframe 
    groups : string
        The name of the column with the groups 
    scores : string
        The name of the column with the scores
    test : string
        to indicate which test to use. Options are 'fisher', 'cochran', 'welch', 'james', 'box', 'scott-smith', 'brown-forsythe', 'alexander-govern', 'mehrotra', 'hartung-agac-makabi', and 'ozdermir-kurt'
    alpha : float between 0 and 1
        Alpha level to be used
    iters : boolean
        to indicate the use of an iteration approach. Only applies to James and Ozdemir-Kurt
    order : integer 0, 1 or 2
        to indicate the James test order. 0 for large sample approximation, 1 for first order and 2 for second order
    alt : boolean
        to indicate the use of an alternative calculation. Only applies to James and Hartung-Agac-Makabi
    
    Returns
    -------
    out : Pandas dataframe
        a dataframe with the test results. This usually has the test-statistic value, degrees of freedom, p-value, name of the test used, a comment if applicable, a boolean to reject the null hypothesis or not (based on the alpha level). In some cases a critical value is shown to compare to the test-statistic.
   
    Notes
    -----
    This function can perform the following tests:

    * Fisher/Classic one-way ANOVA
    * Cochran test for Means
    * Welch one-way ANOVA <span style="color:lightsteelblue">(Welch, 1951, pp. 330, 334, 335)</span>
    * James test (large sample sizes, first-order, and second-order) <span style="color:lightsteelblue">(James, 1951)</span>
    * Box correction for Fisher <span style="color:lightsteelblue">(Box, 1954, p. 299)</span>
    * Scott-Smith test for Means <span style="color:lightsteelblue">(Scott &amp; Smith, 1971, p. 277)</span>
    * Brown-Forsythe test for Means <span style="color:lightsteelblue">(Brown &amp; Forsythe, 1974, p. 130)</span>
    * Alexander-Govern test for Means <span style="color:lightsteelblue">(Alexander &amp; Govern, 1994, pp. 92-94)</span>
    * Mehrotra modified Brown-Forsythe <span style="color:lightsteelblue">(Mehrotra, 1997, p. 1141)</span>
    * Hartung-Agac-Makabi adjusted Welch <span style="color:lightsteelblue">(Hartung et al., 2002, pp. 206-207)</span>
    * Özdemir-Kurt B2 test <span style="color:lightsteelblue">(Özdemir &amp; Kurt, 2006, pp. 85-86)</span>

    Unfortunately my skills (or perhaps it was just energy) lack to fully understand what is supposedly the original article from Cochran  (1937). The formulas shown in this file are based on Cavus and Yazici (2020, p. 5), Hartung et al. (2002, p. 202) and also Mezui-Mbeng (2015, p. 787).

    Each test attempts to test if at least two means in the categories are significantly different. A separate Jupyter Notebook is available for each test with more details.

    In the R library 'doex' a Johansen test (1980) is also available, but this will give the same results as the Welch one-way ANOVA. Also an Asiribo-Gurland test (1990) is in that library, but this gives the same results as the Box correction. The library also mentions an Aspin-Welch (1948, 1949) test, but I suspect the formula there has a small error and it should give the same result as the Welch.

    The second degrees of freedom in the 'doex' library for the Box correction are also calculated different. They were the same as in the R-library 'onewaytests', but this library got updated and changed to the version used in this document, after some communication with the creator. 

    The calculation for $v_j$ values in the second-order James test is also open for debate. Here $v_j = n_j - 2$ is used, while others might use $v_j = n_j - 1$. More details on this in the separate Jupyter Notebook on the James test. In the function it is possible to choose which one to use.

    Similar for the Hartung-Agac-Makabi. Here $\phi_j = \frac{n_j + 2}{n_j + 1}$ is used, while the R-library 'doex' uses $\phi_j = \frac{n_j - 1}{n_j - 3}$. More details on this in the separate Jupyter Notebook on the test. In the function  it is possible to choose which one to use.
    
    ## Formulas
    The following generic symbols are used:

    * $x_{i, j}$ the i-th score in category j
    * $n_j$ the sample size of category j
    * $k$ the number of categories

    ### Re-Usable Parts

    All tests make use of the sample mean ($\bar{x}_j$) and variance ($s_j^2$) per category:

    $$\bar{x}_j = \frac{\sum_{j=1}^{n_j} x_{i,j}}{n_j}$$

    $$s_j^2 = \frac{\sum_{i=1}^{n_j} \left(x_{i,j} - \bar{x}_j\right)^2}{n_j - 1}$$


    Fisher, Box, Scott-Smith, Brown-Forsythe, Alexander-Govern, and Mehrotra also require the total sample size ($n$) and the overall mean ($\bar{x}$):

    $$n = \sum_{j=1}^k n_j$$

    $$\bar{x} = \frac{\sum_{j=1}^{n_j}n_j\times \bar{x}_j}{n} = \frac{\sum_{j=1}^{k}\sum_{i=1}^{n_j} x_{i,j}}{n}$$


    The Cochran Means, the Welch, James, Alexander-Govern, and Özdemir-Kurt B2 test all use weights ($w_j$), adjusted weights ($h_j$) and an overall weighted mean ($y_w$):

    $$w_j = \frac{n_j}{s_j^2}$$

    $$w = \sum_{j=1}^k w_j$$

    $$h_j = \frac{w_j}{w}$$

    $$\bar{y}_w = \sum_{j=1}^k h_j\times \bar{x}_j$$

    Hartung-Agac-Makabi adjust the weights by a factor, which as a consequence also adjusts the $w$, $h_j$, and $\bar{y}_w$:

    $$\phi_j = \frac{n_j + 2}{n_j + 1}$$

    $$w_j^* = \frac{n_j}{s_j^2}\times\frac{1}{\phi_j}$$

    $$h_j^* = \frac{w_j}{w}$$

    $$\bar{y}_w^* = \sum_{j=1}^k h_j^*\times \bar{x}_j$$


    Welch, James, and Hartung-Agac-Makabi also use a lambda ($\lambda$), of course Hartung-Agac-Makabi has the adjusted version ($\lambda^*$) since it uses the weights:

    $$\lambda = \sum_{j=1}^k \frac{\left(1 - h_j\right)^2}{n_j - 1}$$

    $$\lambda^* = \sum_{j=1}^k \frac{\left(1 - h_j^*\right)^2}{n_j - 1}$$

    For James second order test the denominator changes as well:

    $$\lambda_{2} = \sum_{j=1}^k \frac{\left(1 - h_j\right)^2}{n_j - 2}$$

    Cochran, Welch, James, and Hartung-Agac-Makabi use the Cochran chi-square statistic:

    $$\chi_{Cochran}^2=\sum_{j=1}^k w_j\times\left(\bar{x}_j - \bar{y}_w\right)^2$$

    For Hartung-Agac-Makabi they use adjusted weights, so:

    $$C^*=\sum_{j=1}^k w_j^*\times\left(\bar{x}_j - \bar{y}_w^*\right)^2$$

    Scott-Smith, Alexander-Govern, and Ozdemir-Kurt use a t-value. Scott-Smith version uses the unweighted mean ($\bar{x}$), while the other two use the weighted version ($\bar{y}_w$).

    Scott-Smith:
    $$t_j = \frac{\bar{x}_j - \bar{x}}{\sqrt{\frac{s_j^2}{n_j}}}$$

    Alexander-Govern, and Ozdemir-Kurt:
    $$t_j^* = \frac{\bar{x}_j - \bar{y}_w}{\sqrt{\frac{s_j^2}{n_j}}}$$


    ### Test Statistics

    The **Cochran statistic** was already shown earlier, since it is re-used in some other tests. It was defined as:

    $$\chi_{Cochran}^2=\sum_{j=1}^k w_j\times\left(\bar{x}_j - \bar{y}_w\right)^2$$


    The **Welch statistic** uses and adjustment on the Cochran statistic.

    $$W_{adj}= k-1 + \frac{2\times\left(k-2\right)}{k^2+1}\times \lambda$$

    $$F_{Welch} = \frac{\chi_{Cochran}^2}{W_{adj}}$$

    **Hartung-Agac-Makabi** have the exact same approach, but then using their adjusted values:

    $$W_{HAM adj}= k-1 + \frac{2\times\left(k-2\right)}{k^2+1}\times \lambda^*$$

    $$F_{HAM} = \frac{C^*}{W_{HAM adj}}$$

    **Scott and Smith** adjust the t-values to z-values, then sum the squares of those for their test value:

    $$z_j = t_j\times\sqrt{\frac{n_j-3}{n_j-1}}$$

    $$\chi_{SS}^2 = \sum_{j=1}^k z_j^2$$


    **Alexander and Govern** also adjust the t-values to z-values and sum the squares of those, but go about it differently:

    $$a_j = n_j - 1.5$$

    $$b_j = 48\times a_j^2$$

    $$c_j = \sqrt{a_j\times\ln\left(1 + \frac{\left(t_j^*\right)^2}{n_j - 1}\right)}$$

    $$z_j = c_j + \frac{c_j^3 + 3\times c_j}{b_j} - \frac{4\times c_j^7 + 33\times c_j^5 + 240\times c_j^3 + 855\times c_j}{10\times b_j^2 + 8\times b_j\times c_j^4 + 1000\times b_j}$$

    $$\chi_{AG}^2 = \sum_{j=1}^k z_j^2$$

    **Ozdemir and Kurt** use also determine z-values, but have a different approach:

    First a critical z-value is calculated based on a given alpha level. This is the inverse cumulative distribution function, also known as the quantile function or percent-point function:

    $$z_{crit} = Q\left(Z\left(1 - \frac{\alpha}{2}\right)\right)$$

    Then the z-values and test statistic can be calculated:

    $$v_j = n_j - 1$$

    $$c_j = \frac{4\times v_j^2 + \frac{5\times\left(2\times z_{crit}^2+3\right)}{24}}{4\times v_j^2+v_j+\frac{4\times z_{crit}^2+9}{12}}\times\sqrt{v_j}$$

    $$z = c_j\times\sqrt{\ln\left(1+\frac{\left(t_j^*\right)^2}{v_j}\right)}$$

    $$\chi_{OK}^2 = \sum_{j=1}^k z_j^2$$

    The **Fisher one-way anova** uses:

    $$F = \frac{\left(n - k\right)\times\sum_{j=1}^{k}n_j\times(\bar{x}_j-\bar{x})^2}{\left(k - 1\right)\times\sum_{j=1}^{k}\sum_{i=1}^{n_j}(x_{i,j}-\bar{x}_j)^2}$$


    **Box** uses a correction on this and the test statistic is the same as for the **Brown-Forsythe** and **Mehrotra**:

    $$B_{adj} = \frac{n-k}{n\times\left(k-1\right)}\times\frac{\sum_{j=1}^k\left(n-n_j\right)\times s_j^2}{\sum_{j=1}^k\left(n_j-1\right)\times s_j^2}$$

    $$F_{Box} = F_{BF} = F_{Mehrotra} =\frac{F}{B_{adj}}$$

    ### Degrees of Freedom

    #### (First) Degrees of Freedom

    The (first) degrees of freedom for the **Fisher**, **Cochran**, **Welch**, **James**, **Brown-Forsythe**, **Alexander-Govern**, **Hartung-Agac-Makabi**, and **Ozdemir-Kurt** are all the same:

    $$df_1 = k - 1$$

    For **Box** and **Mehrotra** the first degrees of freedom are also the same:

    $$df_1 = \frac{\left(\sum_{j=1}^k\left(n-n_j\right)\times s_j^2\right)^2}{\left(\sum_{j=1}^k n_j\times s_j^2\right)^2 + n\times\sum_{j=1}^k\left(n - 2\times n_j\right)\times s_j^4}$$


    For **Scott-Smith** the degrees of freedom are:

    $$df = k$$

    #### Second Degrees of Freedom

    For tests based on the F-distribution the second degrees of freedom vary quite a bit.

    **Fisher** one-way ANOVA:

    $$df_2 = n - k$$

    **Welch**:

    $$df_2 = \frac{k^2-1}{3\times\lambda}$$

    **Hartung-Agac-Makabi** similar to Welch, but using their adjusted lambda:

    $$df_2 = \frac{k^2-1}{3\times\lambda^*}$$

    **Box**:

    $$df_2 = \frac{\left(\sum_{j=1}^k \left(n_j-1\right)\times s_j^2\right)^2}{\sum_{j=1}^k\left(n_j-1\right)\times s_j^4}$$

    **Brown-Forsythe** and **Mehrotra**:

    $$df_2 = \frac{\left(\sum_{j=1}^k\left(1-\frac{n_j}{n}\right)\times s_j^2\right)^2}{\sum_{j=1}^k \frac{\left(1-\frac{n_j}{n}\right)^2\times s_j^4}{n_j - 1}}$$

    ## p-values

    The p-value of chi-square based tests can be calculated using the chi-square distribution:

    $$\chi_{Cochran}^2 \sim \chi^2\left(df\right)$$

    $$\chi_{SS}^2 \sim \chi^2\left(df\right)$$

    $$\chi_{AG}^2 \sim \chi^2\left(df\right)$$

    $$\chi_{OK}^2 \sim \chi^2\left(df\right)$$


    The F based tests, of course with the F-distribution:

    $$F \sim F\left(df_1, df_2\right)$$

    $$F_{Welch} \sim F\left(df_1, df_2\right)$$

    $$F_{Box} \sim F\left(df_1, df_2\right)$$

    $$F_{BF} \sim F\left(df_1, df_2\right)$$

    $$F_{Mehrotra} \sim F\left(df_1, df_2\right)$$

    $$F_{HAM} \sim F\left(df_1, df_2\right)$$


    The James and Ozdemir and Kurt test are a bit special. They both require a critical chi-square value based on a given alpha level:

    $$\chi_{crit}^2 = Q\left(\chi^2\left(1 - \alpha, df\right)\right)$$


    The **Ozdemir and Kurt** test compares the critical chi-square value to the calculated $$\chi_{OK}^2$$ value.

    If $\chi_{OK}^2 > \chi_{crit}^2$ reject the null hypothesis (i.e. p < alpha)

    For the **James** test a critical J-statistic is calculated. There are two versions for this:

    First order James:

    $$J_{crit} = \chi_{crit}^2\times\left(1 + \frac{3\times \chi_{crit}^2 + k + 1}{2\times\left(k^2 - 1\right)}\times\lambda\right)$$

    Second order James:

    $$R_{xy} = \sum_{j=1}^k \frac{h_j^y}{\left(n_j - 2\right)^x}$$

    $$\chi_{2\times r} = \frac{\left(\chi_{crit}^2\right)^r}{\prod_{i=1}^r \left(k + 2\times i - 3\right)}$$

    $$a_1 = \chi_{crit}^2 + \frac{1}{2}\times\left(3\times\chi_4+\chi_2\right)\times\lambda_2$$
    $$a_2 = \frac{1}{16}\times\left(3\times\chi_4+\chi_2\right)^2\times\left(1-\frac{k-3}{\chi_{crit}^2}\right)\times\lambda_2^2$$
    $$a_{3f} = \frac{1}{2}\times\left(3\times\chi_4+\chi_2\right)$$
    $$a_{3a} = 8\times R_{23} - 10\times R_{22} + 4\times R_{21} - 6\times R_{12}^2 + 8\times R_{12}\times R_{11} - 4\times R_{11}^2$$
    $$a_{3b} = \left(2\times R_{23} - 4\times R_{22} + 2\times R_{21} - 2\times R_{12}^2 + 4\times R_{12}\times R_{11} - 2\times R_{11}^2\right)\times\left(\chi_2-1\right)$$
    $$a_{3c} = \frac{1}{4}\times\left(-R_{12}^2 + 4\times R_{12}\times R_{11} - 2\times R_{12}\times R_{10} - 4\times R_{11}^2 + 4\times R_{11}\times R_{10} - R_{10}^2\right)\times\left(3\times\chi_4 - 2\times\chi_2 - 1\right)$$
    $$a_3 = a_{3f}\times\left(a_{3a} + a_{3b} + a_{3c} \right)$$
    $$a_4 = \left(R_{23} - 3\times R_{22} + 3\times R_{21} - R_{20}\right)\times\left(5\times \chi_6 + 2\times\chi_4 + \chi_2\right)$$
    $$a_5 = \frac{3}{16}\times\left(R_{12}^2 - 4\times R_{23} + 6\times R_{22} - 4\times R_{21} + R_{20}\right)\times\left(35\times\chi_8 + 15\times\chi_6 + 9\times\chi_4 + 5\times\chi_2\right)$$
    $$a_6 = \frac{1}{16}\times\left(-2\times R_{22}^2 + 4\times R_{21} - R_{20} + 2\times R_{12}\times R_{10} - 4\times R_{11}\times R_{10} + R_{10}^2\right)\times\left(9\times\chi_8 - 3\times\chi_6 - 5\times\chi_4 - \chi_2\right)$$ 
    $$a_7 = \frac{1}{16}\times\left(-2\times R_{22}^2 + 4\times R_{21} - R_20 + 2\times R_{12}\times R_{10} - 4\times R_{11}\times R_{10} + R_{10}^2\right)\times\left(9\times\chi_8 - 3\times\chi_6 - 5\times\chi_4 - \chi_2\right)$$
    $$a_8 = \frac{1}{4}\times\left(-R_{22} + R_{11}^2\right)\times\left(27\times\chi_8 + 3\times\chi_6 + \chi_4 + \chi_2\right)$$
    $$a_9 = \frac{1}{4}\times\left(R_{23} - R_{12}\times R_{11}\right)\times\left(45\times\chi_8 + 9\times\chi_6 + 7\times\chi_4 + 3\times\chi_2\right)$$

    $$J_{crit} = \sum_{r=1}^9 a_r$$

    The critical J value is then compared to the J-statistic. If $J > J_{crit}$ reject the null hypothesis (i.e. p < alpha)

    Note: For both the James and Ozdemir-Kurt test the function can do an iteration to find a p-value such that the test statistic will equal the critical value. I suspect this might give a better approximate p-value.
    
    Examples
    ---------
    >>> scores = [20, 50, 80, 15, 40, 85, 30, 45, 70, 60, 90, 25, 40, 70, 65, 70, 98, 40, 65, 
            60, 35, 50, 40, 75, 65, 70, 20, 80, 35, 68, 70, 60, 70, 80, 98, 10, 40, 63, 
            75, 80, 40, 90, 100, 33, 36, 65, 78, 50]
    >>> categ = ["Rotterdam", "Haarlem", "Diemen", "Rotterdam", "Haarlem", "Diemen", 
           "Rotterdam", "Haarlem", "Diemen", "Rotterdam", "Diemen", "Rotterdam", 
           "Haarlem", "Diemen", "Rotterdam", "Diemen", "Rotterdam", "Haarlem", 
           "Diemen", "Rotterdam", "Haarlem", "Rotterdam", "Haarlem", "Diemen", 
           "Haarlem", "Diemen", "Haarlem", "Diemen", "Rotterdam", "Diemen", 
           "Rotterdam", "Haarlem", "Diemen", "Haarlem", "Diemen", "Rotterdam", 
           "Haarlem", "Diemen", "Rotterdam", "Haarlem", "Diemen", "Haarlem", 
           "Diemen", "Haarlem", "Haarlem", "Haarlem", "Haarlem", "Haarlem"]
    >>> df = pd.DataFrame()
    >>> df['Location'] = categ
    >>> df['Over_Grade'] = scores
    >>> meansTest(df, 'Location', 'Over_Grade', test='welch')
    >>> meansTest(df, 'Location', 'Over_Grade', test='james', order=1, iters=True)
    
    References
    ----------
    
    Alexander, R. A., & Govern, D. M. (1994). A new and simpler approximation for ANOVA under variance heterogeneity. *Journal of Educational Statistics, 19*(2), 91–101. https://doi.org/10.2307/1165140

    Asiribo, O., & Gurland, J. (1990). Coping with variance heterogeneity. *Communications in Statistics - Theory and Methods, 19*(11), 4029–4048. https://doi.org/10.1080/03610929008830427

    Aspin, A. A. (1948). An examination and further development of a formula arising in the problem of comparing two mean values. *Biometrika, 35*(1/2), 88–96. https://doi.org/10.2307/2332631

    Aspin, A. A., & Welch, B. L. (1949). Tables for use in comparisons whose accuracy involves two variances, separately estimated. *Biometrika, 36*(3/4), 290. https://doi.org/10.2307/2332668

    Box, G. E. P. (1954). Some theorems on quadratic forms applied in the study of analysis of variance problems, I: Effect of inequality of variance in the one-way classification. *The Annals of Mathematical Statistics, 25*(2), 290–302. https://doi.org/10.1214/aoms/1177728786

    Brown, M. B., & Forsythe, A. B. (1974). The small sample behavior of some statistics which test the equality of several means. *Technometrics, 16*(1), 129–132. https://doi.org/10.1080/00401706.1974.10489158

    Cavus, M., & Yazıcı, B. (2020). Testing the equality of normal distributed and independent groups’ means under unequal variances by doex package. *The R Journal, 12*(2), 134. https://doi.org/10.32614/RJ-2021-008

    Hartung, J., Argaç, D., & Makambi, K. H. (2002). Small sample properties of tests on homogeneity in one-way anova and meta-analysis. *Statistical Papers, 43*(2), 197–235. https://doi.org/10.1007/s00362-002-0097-8

    James, G. S. (1951). The comparison of several groups of observations when the ratios of the population variances are unknown. *Biometrika, 38*(3–4), 324–329. https://doi.org/10.1093/biomet/38.3-4.324

    Johansen, S. (1980). The Welch-James approximation to the distribution of the residual sum of squares in a weighted linear regression. *Biometrika, 67*(1), 85–92. https://doi.org/10.1093/biomet/67.1.85

    Mehrotra, D. V. (1997). Improving the Brown-Forsythe solution to the generalized Behrens-Fisher problem. *Communications in Statistics - Simulation and Computation, 26*(3), 1139–1145. https://doi.org/10.1080/03610919708813431

    Mezui-Mbeng, P. (2015). A note on Cochran test for homogeneity in two ways ANOVA and meta-analysis. *Open Journal of Statistics, 5**(7), 787–796. https://doi.org/10.4236/ojs.2015.57078

    Özdemir, A. F., & Kurt, S. (2006). One way fixed effect analysis of variance under variance heterogeneity and a solution proposal. *Selçuk Journal of Applied Mathematics, 7*(2), 81–90.

    Scott, A. J., & Smith, T. M. F. (1971). Interval estimates for linear combinations of means. *Applied Statistics, 20*(3), 276–285. https://doi.org/10.2307/2346757

    Welch, B. L. (1951). On the comparison of several mean values: An alternative approach. *Biometrika, 38*(3/4), 330. https://doi.org/10.2307/2332579
    
    Author
    ------
    Made by P. Stikker
    
    Please visit: https://PeterStatistics.com
    
    YouTube channel: https://www.youtube.com/stikpet
    
    '''
    
    names = {'fisher' : "Fisher one-way anova", 
             'cochran': "Cochran for Means",
             'welch' : "Welch one-way anova", 
             'james' : "James test", 
             'box' : "Box correction for Fisher", 
             'scott-smith': "Scott and Smith", 
             'brown-forsythe' : "Brown-Forsythe for Means", 
             'alexander-govern' : "Alexander-Govern", 
             'mehrotra' : "Mehrotra modified Brown-Forsythe", 
             'hartung-agac-makabi' : "Hartung-Agac-Makabi adjusted Welch", 
             'ozdemir-kurt' : "Özdermir-Kurt B2"}
    
    comment = None
    
    #sample size, mean, and variance per category
    nj = data.groupby(groups)[scores].count()
    meanj = data.groupby(groups)[scores].mean()
    varj = data.groupby(groups)[scores].var()
    res = pd.concat([nj, meanj, varj], axis=1)
    res = res.set_axis(["n", "mean", "var"], axis=1, inplace=False)
    
    #number of categories
    k = len(res)
    
    #(first) degrees of freedom for many tests
    if test=='fisher' or test=='box' or test=='cochran' or test=='welch' or test=='james' or test=='brown-forsythe' or test=='alexander-govern' or test=='hartung-agac-makabi' or test=='ozdemir-kurt':
        df1 = k - 1
    
    #weights (w), adjusted weights (h), weighted mean, and cochran test statistic
    if test=='cochran' or test=='welch' or test=='james' or test=='alexander-govern' or test=='ozdemir-kurt' or test=='hartung-agac-makabi':
        
        if test=='hartung-agac-makabi':
            if alt:
                res['phi'] = (res['n'] - 1)/(res['n'] - 3)
            else:
                res['phi'] = (res['n'] + 2)/(res['n'] + 1)
            res['w'] = res['n']/res['var'] * 1/res['phi']
        else:
            res['w'] = res['n']/res['var']
        w = sum(res['w'])
        res['h'] = res['w']/w
        
        yw = sum(res['h']*res['mean'])
        
        chi2Cochran = sum(res['w']*(res['mean'] - yw)**2)
        
        if test=='cochran':
            chi2Stat = chi2Cochran
    
    #lambda
    if test=='welch' or test=='james' or test=='hartung-agac-makabi':
        lamb = sum((1 - res['h'])**2/(res['n'] - 1))
        if test=='welch' or test=='hartung-agac-makabi':
            Fstat = chi2Cochran / (k - 1 + 2*(k - 2)/(k + 1)*lamb)
            df2 = (k**2 - 1)/(3*lamb)
    
    #overall sample size and mean
    if test=='fisher' or test=='box' or test=='scott-smith' or test=='brown-forsythe' or test=='alexander-govern' or test=='mehrotra' or test=='ozdemir-kurt':
        n = sum(res['n'])
        mean = sum(res['n']*res['mean'])/n
    
    #the fisher test and box correction
    if test=='fisher' or test=='box':
        ssb = sum(res['n']*(res['mean'] - mean)**2)
        ssw = variance(data[scores])*(n - 1) - ssb
        df2 = n - k
        Fstat = (ssb/df1)/(ssw/df2)
        
        if test=='box':
            c = (n - k)/(n*(k - 1))*sum((n - res['n'])*res['var']) / sum((res['n'] - 1)*res['var'])
            Fstat = Fstat/c
            df1 = sum((n - res['n'])*res['var'])**2 / (sum(res['n']*res['var'])**2 + n*sum((n - 2*res['n'])*res['var']**2))
            df2 = sum((res['n'] - 1)*res['var'])**2 / (sum((res['n'] - 1)*res['var']**2))
    
    #t-values
    if test=='scott-smith' or test=='alexander-govern' or test=='ozdemir-kurt':
        if test=='ozdemir-kurt' or test=='alexander-govern':
            mean = yw
        res['t'] = (res['mean'] - mean)/(res['var']/res['n'])**0.5
    
    #scott-smith test
    if test=='scott-smith':
        chi2Stat = sum((res['t']*((res['n'] - 3)/(res['n'] - 1))**0.5)**2)
        df1 = k
    
    #brown-forsythe and mehrotra's adjustment
    if test=='brown-forsythe' or test=='mehrotra':
        Fstat = sum(res["n"]*(res["mean"] - mean)**2)/sum((1 - res["n"]/n)*res["var"])
        df2 = sum((1 - res["n"]/n)*res["var"])**2/sum((1 - res['n']/n)**2*res['var']**2/(res['n'] - 1))
        
        if test=='mehrotra':
            df1 = (sum(res['var']) - sum(res['n']*res['var'])/n)**2 / (sum(res['var']**2) + (sum(res['n']*res['var'])/n)**2 - 2*sum(res['n']*res['var']**2)/n)
    
    #alexander-govern test
    if test=='alexander-govern':
        res['a'] = res['n'] - 1.5
        res['b'] = 48*res['a']**2
        res['c'] = (res['a']*log(1 + res['t']**2/(res['n'] - 1)))**0.5
        res['z'] = res['c'] + (res['c']**3 + 3*res['c'])/res['b'] - (4*res['c']**7 + 33*res['c']**5 + 240*res['c']**3 + 855*res['c'])/(10*res['b']**2 + 8*res['b']*res['c']**4 + 1000*res['b'])
        chi2Stat = sum(res['z']**2)
    
    #ozdermir-kurt test
    if test=='ozdemir-kurt':
        res['v'] = res['n'] - 1
        zCrit = NormalDist().inv_cdf(1-alpha/2)
        res['c'] = (4*res['v']**2 + 5*(2*zCrit**2 + 3)/24)/(4*res['v']**2 + res['v'] + (4*zCrit**2 + 9)/12) * res['v']**0.5
        res['z'] = res['c']*(log(1 + res['t']**2/res['v']))**0.5
        chi2Stat = sum(res['z']**2)
        
        if iters:
            comment = "using iterations for approximating p-value"
            df = k - 1
            pLow = 0
            pHigh = 1
            pVal = 0.05
            nIter = 1
            whileDo = True

            while whileDo:
                zCrit = NormalDist().inv_cdf(1-pVal/2)
                #the c-values and z-values
                res['c'] = (4*res['v']**2 + 5*(2*zCrit**2 + 3)/24)/(4*res['v']**2 + res['v'] + (4*zCrit**2 + 9)/12) * res['v']**0.5
                res['z'] = res['c']*(log(1 + res['t']**2/res['v']))**0.5
                chi2Stat = sum(res['z']**2)
                chi2Crit = chi2.ppf(1-pVal, df)

                if chi2Crit < chi2Stat:
                    pHigh = pVal
                    pVal = (pLow + pVal)/2
                elif chi2Crit > chi2Stat:
                    pLow = pVal
                    pVal = (pHigh + pVal)/2

                nIter = nIter + 1

                if chi2Crit == chi2Stat or nIter >= 500:
                    whileDo = False
    
    #james test
    if test=='james':
        J = chi2Cochran
        
        if order==0:
            chi2Stat = J
            pVal = chi2.sf(J, df1) 
            reject = pVal < alpha
            comment = "for large category sizes"
            out = pd.DataFrame([[J, df1, pVal, reject]], columns=["statistic", "df1", "p-value", "reject H0"])
            
        else:
            cCrit = chi2.ppf(1-alpha, df1)
            
            if order==1:
                Jcrit = cCrit*(1 + (3*cCrit  + k + 1)/(2*(k**2 - 1))*lamb)
                
                if iters:
                    comment = "first-order with iterations for p-value approximation"
                    pLow = 0
                    pHigh = 1
                    pVal = 0.05
                    nIter = 1
                    whileDo = True
                    
                    while whileDo:
                        cCrit = chi2.ppf(1-pVal, df1)
                        Jcrit = cCrit*(1 + (3*cCrit  + k + 1)/(2*(k**2 - 1))*lamb)
                        if Jcrit < J:
                            pHigh = pVal
                            pVal = (pLow + pVal)/2
                        elif Jcrit > J:
                            pLow = pVal
                            pVal = (pHigh + pVal)/2

                        nIter = nIter + 1

                        if Jcrit == J or nIter >= 800:
                            whileDo = False
                            
                    reject = pVal < alpha
                    out = pd.DataFrame([[J, df1, pVal, reject]], columns=["statistic", "df1", "p-value", "reject H0"])
                            
                else:
                    comment = "first-order"
                    reject = J > Jcrit
                    out = pd.DataFrame([[J, df1, Jcrit, reject]], columns=["statistic", "df1", "J-critical", "reject H0"])
                            
                            
            else:
                if not(alt):
                    comment = "second order"
                    res['v'] = res['n'] - 2
                    lamb = sum((1 - res['h'])**2/res['v'])
                else:
                    comment = "second order with alternative v (v = n -1)"
                    res['v'] = res['n'] - 1
                    
                R10 = sum(res['h']**0 / res['v']**1)
                R11 = sum(res['h']**1 / res['v']**1)
                R12 = sum(res['h']**2 / res['v']**1)
                R20 = sum(res['h']**0 / res['v']**2)
                R21 = sum(res['h']**1 / res['v']**2)
                R22 = sum(res['h']**2 / res['v']**2)
                R23 = sum(res['h']**3 / res['v']**2)
                
                c2 = cCrit**1/(k + 2*1 - 3)
                c4 = c2 * cCrit/(k + 2*2 - 3)
                c6 = c4 * cCrit/(k + 2*3 - 3)
                c8 = c6 * cCrit/(k + 2*4 - 3)
                
                Jcrit = cCrit + 1/2*(3*c4+c2)*lamb + \
                1/16*(3*c4 + c2 )**2*(1-(k-3)/cCrit)*lamb**2 + \
                1/2*(3*c4 + c2 )*\
                ((8*R23 - 10*R22 + 4*R21 - 6*R12**2 + 8*R12*R11 - 4*R11**2) + \
                 (2*R23 - 4*R22 + 2*R21 - 2*R12**2 + 4*R12*R11 - 2*R11**2)*(c2 - 1) + \
                 1/4*(-R12**2 + 4*R12*R11 - 2*R12*R10 - 4*R11**2 + 4*R11*R10 - R10**2 )*(3*c4 - 2*c2 - 1)) + \
                (R23 - 3*R22 + 3*R21 - R20)*(5*c6 + 2*c4 + c2) + \
                3/16*(R12**2 - 4*R23 + 6*R22 - 4*R21 + R20)*(35*c8 + 15*c6 + 9*c4 + 5*c2) + \
                1/16*(-2*R22**2 + 4*R21 - R20 + 2*R12*R10 - 4*R11*R10 + R10**2)*(9*c8 - 3*c6 - 5*c4 - c2) + \
                1/4*(-R22 + R11**2 )*(27*c8 + 3*c6 + c4 + c2) + \
                1/4*(R23 - R12*R11)*(45*c8 + 9*c6 + 7*c4 + 3*c2)
                
                if iters:
                    comment = comment + ", using iterations for p-value approximation"
                    pLow = 0
                    pHigh = 1
                    pVal = 0.05
                    nIter = 1
                    whileDo = True
                    
                    while whileDo:
                        cCrit = chi2.ppf(1-pVal, df1)

                        #(re)calculate chi values
                        c2 = cCrit**1/(k + 2*1 - 3)
                        c4 = c2 * cCrit/(k + 2*2 - 3)
                        c6 = c4 * cCrit/(k + 2*3 - 3)
                        c8 = c6 * cCrit/(k + 2*4 - 3)

                        #calculate Jcrit
                        Jcrit = cCrit + 1/2*(3*c4+c2)*lamb + \
                        1/16*(3*c4 + c2 )**2*(1-(k-3)/cCrit)*lamb**2 + \
                        1/2*(3*c4 + c2 )*\
                        ((8*R23 - 10*R22 + 4*R21 - 6*R12**2 + 8*R12*R11 - 4*R11**2) + \
                         (2*R23 - 4*R22 + 2*R21 - 2*R12**2 + 4*R12*R11 - 2*R11**2)*(c2 - 1) + \
                         1/4*(-R12**2 + 4*R12*R11 - 2*R12*R10 - 4*R11**2 + 4*R11*R10 - R10**2 )*(3*c4 - 2*c2 - 1)) + \
                        (R23 - 3*R22 + 3*R21 - R20)*(5*c6 + 2*c4 + c2) + \
                        3/16*(R12**2 - 4*R23 + 6*R22 - 4*R21 + R20)*(35*c8 + 15*c6 + 9*c4 + 5*c2) + \
                        1/16*(-2*R22**2 + 4*R21 - R20 + 2*R12*R10 - 4*R11*R10 + R10**2)*(9*c8 - 3*c6 - 5*c4 - c2) + \
                        1/4*(-R22 + R11**2 )*(27*c8 + 3*c6 + c4 + c2) + \
                        1/4*(R23 - R12*R11)*(45*c8 + 9*c6 + 7*c4 + 3*c2)

                        if Jcrit < J:
                            pHigh = pVal
                            pVal = (pLow + pVal)/2
                        elif Jcrit > J:
                            pLow = pVal
                            pVal = (pHigh + pVal)/2

                        nIter = nIter + 1

                        if Jcrit == J or nIter >= 500:
                            whileDo = False
                    
                    reject = pVal < alpha
                    out = pd.DataFrame([[J, df1, pVal, reject]], columns=["statistic", "df1", "p-value", "reject H0"])
                    
                else:
                    reject = J > Jcrit
                    out = pd.DataFrame([[J, df1, Jcrit, reject]], columns=["statistic", "df1", "J-critical", "reject H0"])
    
    #p-values for F-distribution tests
    if test=='fisher' or test=='box' or test=='welch' or test=='box' or test=='brown-forsythe' or test=='mehrotra' or test=='hartung-agac-makabi':
        pVal = f.sf(Fstat, df1, df2)
        reject = pVal < alpha
        out = pd.DataFrame([[Fstat, df1, df2, pVal, reject]], columns=["statistic", "df1", "df2", "p-value", "reject H0"])
    
    #p-value for chi-square distribution tests
    if test=='cochran' or test=='scott-smith' or test=='alexander-govern' or test=='ozdemir-kurt':
        pVal = chi2.sf(chi2Stat, df1)
        reject = pVal < alpha
        out = pd.DataFrame([[chi2Stat, df1, pVal, reject]], columns=["statistic", "df1", "p-value", "reject H0"])
    
    name = names[test]
    out['test'] = name
    out['comment'] = comment
    
    return out