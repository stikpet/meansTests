#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#  ----MEANS TESTS----                                                       #
#Created by P. Stikker.                                                      #  
#companion website at https://PeterStatistics.com                            #
#YouTube channel: https://youtube.com/stikpet                                #
#Donations welcome via Patreon: https://www.patreon.com/bePatron?u=19398076  #  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#A function that can perform various one-way anovas (test to compare means)

meansTest <- function(scores, groups, test, alpha=0.05, iters=FALSE, order=2, alt=FALSE){
  
  names = c('fisher' = "Fisher one-way anova", 
            'cochran'= "Cochran for Means",
            'welch' = "Welch one-way anova", 
            'james' = "James test", 
            'box' = "Box correction for Fisher", 
            'scott-smith'= "Scott and Smith", 
            'brown-forsythe' = "Brown-Forsythe for Means", 
            'alexander-govern' = "Alexander-Govern", 
            'mehrotra' = "Mehrotra modified Brown-Forsythe", 
            'hartung-agac-makabi' = "Hartung-Agac-Makabi adjusted Welch", 
            'ozdemir-kurt' = "Özdermir-Kurt B2")
  
  comment = NULL
  #sample size, mean, and variance per category
  counts <- setNames(aggregate(scores~groups, FUN=length), c("category", "n"))
  means <- setNames(aggregate(scores~groups, FUN=mean), c("category", "mean"))
  vars <- setNames(aggregate(scores~groups, FUN=var), c("category", "var"))
  res <- merge(counts, means, by = 'category')
  res <- merge(res, vars, by = 'category')
  
  #number of categories
  k <- dim(res)[1]
  
  #(first) degrees of freedom for many tests
  if (test=='fisher' | test=='box' | test=='cochran' | test=='welch' | 
      test=='james' | test=='brown-forsythe' | test=='alexander-govern' | 
      test=='hartung-agac-makabi' | test=='ozdemir-kurt'){
    df1 = k - 1
  }
  
  #weights (w), adjusted weights (h), weighted mean, and cochran test statistic
  if (test=='cochran' | test=='welch' | test=='james' | test=='alexander-govern' |
      test=='ozdemir-kurt' | test=='hartung-agac-makabi'){
    
    if (test=='hartung-agac-makabi'){
      if (alt){
        res$phi = (res$n - 1)/(res$n - 3)}
      else{
        res$phi = (res$n + 2)/(res$n + 1)}
      
      res$w = res$n/res$var * 1/res$phi}
    else{
      res$w = res$n/res$var}
    
    w = sum(res$w)
    res$h = res$w/w
    yw = sum(res$h*res$mean)
    chi2Cochran = sum(res$w*(res$mean - yw)**2)
    
    if (test=='cochran'){
      chi2Stat = chi2Cochran}
  }
  
  #lambda
  if (test=='welch' | test=='james' | test=='hartung-agac-makabi'){
    lamb = sum((1 - res$h)**2/(res$n - 1))
    if (test=='welch' | test=='hartung-agac-makabi'){
      Fstat = chi2Cochran / (k - 1 + 2*(k - 2)/(k + 1)*lamb)
      df2 = (k**2 - 1)/(3*lamb)}
  }
  
  #overall sample size and mean
  if (test=='fisher' | test=='box' | test=='scott-smith' | 
      test=='brown-forsythe' | test=='alexander-govern' | 
      test=='mehrotra' | test=='ozdemir-kurt'){
    n = sum(res$n)
    mean = sum(res$n*res$mean)/n}
  
  #the fisher test and box correction
  if (test=='fisher' | test=='box'){
    ssb = sum(res$n*(res$mean - mean)**2)
    ssw = var(scores)*(n - 1) - ssb
    df2 = n - k
    Fstat = (ssb/df1)/(ssw/df2)
    
    if (test=='box'){
      c = (n - k)/(n*(k - 1))*sum((n - res$n)*res$var) / sum((res$n - 1)*res$var)
      Fstat = Fstat/c
      df1 = sum((n - res$n)*res$var)**2 / (sum(res$n*res$var)**2 + n*sum((n - 2*res$n)*res$var**2))
      df2 = sum((res$n - 1)*res$var)**2 / (sum((res$n - 1)*res$var**2))}
  }
  
  #t-values
  if (test=='scott-smith' | test=='alexander-govern' | test=='ozdemir-kurt'){
    if (test=='ozdemir-kurt' | test=='alexander-govern'){
      mean = yw}
    res$t = (res$mean - mean)/(res$var/res$n)**0.5}
  
  #scott-smith test
  if (test=='scott-smith'){
    chi2Stat = sum((res$t*((res$n - 3)/(res$n - 1))**0.5)**2)
    df1 = k}
  
  #brown-forsythe and mehrotra's adjustment
  if (test=='brown-forsythe' | test=='mehrotra'){
    Fstat = sum(res$n*(res$mean - mean)**2)/sum((1 - res$n/n)*res$var)
    df2 = sum((1 - res$n/n)*res$var)**2/sum((1 - res$n/n)**2*res$var**2/(res$n - 1))}
  
  if (test=='mehrotra'){
    df1 = (sum(res$var) - sum(res$n*res$var)/n)**2 / (sum(res$var**2) + (sum(res$n*res$var)/n)**2 - 2*sum(res$n*res$var**2)/n)}
  
  #alexander-govern test
  if (test=='alexander-govern'){
    res$a = res$n - 1.5
    res$b = 48*res$a**2
    res$c = (res$a*log(1 + res$t**2/(res$n - 1)))**0.5
    res$z = res$c + (res$c**3 + 3*res$c)/res$b - (4*res$c**7 + 33*res$c**5 + 240*res$c**3 + 855*res$c)/(10*res$b**2 + 8*res$b*res$c**4 + 1000*res$b)
    chi2Stat = sum(res$z**2)}
  
  #ozdermir-kurt test
  if (test=='ozdemir-kurt'){
    res$v = res$n - 1
    zCrit = qnorm(1-alpha/2)
    res$c = (4*res$v**2 + 5*(2*zCrit**2 + 3)/24)/(4*res$v**2 + res$v + (4*zCrit**2 + 9)/12) * res$v**0.5
    res$z = res$c*(log(1 + res$t**2/res$v))**0.5
    chi2Stat = sum(res$z**2)
    
    if (iters){
      comment = "using iterations for approximating p-value"
      df = k - 1
      pLow = 0
      pHigh = 1
      pVal = 0.05
      nIter = 1
      whileDo = TRUE
      
      while (whileDo){
        zCrit = qnorm(1-pVal/2)
        
        #the c-values and z-values
        res$c = (4*res$v**2 + 5*(2*zCrit**2 + 3)/24)/(4*res$v**2 + res$v + (4*zCrit**2 + 9)/12) * res$v**0.5
        res$z = res$c*(log(1 + res$t**2/res$v))**0.5
        
        chi2Stat = sum(res$z**2)
        chi2Crit = qchisq(pVal, df, lower.tail=FALSE)
        
        if (chi2Crit < chi2Stat){
          pHigh = pVal
          pVal = (pLow + pVal)/2}
        else if (chi2Crit > chi2Stat){
          pLow = pVal
          pVal = (pHigh + pVal)/2}
        
        nIter = nIter + 1
        
        if (chi2Crit == chi2Stat | nIter >= 500){
          whileDo = FALSE}
      }
    }
  }
  
  #james test
  if (test=='james'){
    J = chi2Cochran
    
    if (order==0){
      chi2Stat = J
      pVal = pchisq(J, df1, lower.tail=FALSE) 
      reject = pVal < alpha
      comment = "for large category sizes"
      
      out <- data.frame(J, df1, pVal, reject)
      colnames(out)<-c("statistic", "df1", "p-value", "reject H0")}
    
    else{
      cCrit = qchisq(1-alpha, df1)
      
      if (order==1){
        Jcrit = cCrit*(1 + (3*cCrit  + k + 1)/(2*(k**2 - 1))*lamb)
        
        if (iters){
          comment = "first-order with iterations for p-value approximation"
          pLow = 0
          pHigh = 1
          pVal = 0.05
          nIter = 1
          whileDo = TRUE
          
          while (whileDo){
            cCrit = qchisq(1-pVal, df1)
            Jcrit = cCrit*(1 + (3*cCrit  + k + 1)/(2*(k**2 - 1))*lamb)
            
            if (Jcrit < J){
              pHigh = pVal
              pVal = (pLow + pVal)/2}
            else if (Jcrit > J){
              pLow = pVal
              pVal = (pHigh + pVal)/2}
            
            nIter = nIter + 1
            
            if (Jcrit == J | nIter >= 800){
              whileDo = FALSE}
            }
          
          reject = pVal < alpha
          
          out <- data.frame(J, df1, pVal, reject)
          colnames(out)<-c("statistic", "df1", "p-value", "reject H0")}
        
        else{
          comment = "first-order"
          reject = J > Jcrit
          
          out <- data.frame(J, df1, Jcrit, reject)
          colnames(out)<-c("statistic", "df1", "J-critical", "reject H0")}
        }
      
      else{
        if (!alt){
          comment = "second order"
          res$v = res$n - 2
          lamb = sum((1 - res$h)**2/res$v)}
        
        else{
          comment = "second order with alternative v (v = n -1)"
          res$v = res$n - 1}
        
        R10 = sum(res$h**0 / res$v**1)
        R11 = sum(res$h**1 / res$v**1)
        R12 = sum(res$h**2 / res$v**1)
        R20 = sum(res$h**0 / res$v**2)
        R21 = sum(res$h**1 / res$v**2)
        R22 = sum(res$h**2 / res$v**2)
        R23 = sum(res$h**3 / res$v**2)
        
        c2 = cCrit**1/(k + 2*1 - 3)
        c4 = c2 * cCrit/(k + 2*2 - 3)
        c6 = c4 * cCrit/(k + 2*3 - 3)
        c8 = c6 * cCrit/(k + 2*4 - 3)
        
        Jcrit = cCrit + 1/2*(3*c4+c2)*lamb + 
          1/16*(3*c4 + c2 )**2*(1-(k-3)/cCrit)*lamb**2 + 
          1/2*(3*c4 + c2 )*
          ((8*R23 - 10*R22 + 4*R21 - 6*R12**2 + 8*R12*R11 - 4*R11**2) + 
             (2*R23 - 4*R22 + 2*R21 - 2*R12**2 + 4*R12*R11 - 2*R11**2)*(c2 - 1) + 
             1/4*(-R12**2 + 4*R12*R11 - 2*R12*R10 - 4*R11**2 + 4*R11*R10 - R10**2 )*(3*c4 - 2*c2 - 1)) + 
          (R23 - 3*R22 + 3*R21 - R20)*(5*c6 + 2*c4 + c2) + 
          3/16*(R12**2 - 4*R23 + 6*R22 - 4*R21 + R20)*(35*c8 + 15*c6 + 9*c4 + 5*c2) + 
          1/16*(-2*R22**2 + 4*R21 - R20 + 2*R12*R10 - 4*R11*R10 + R10**2)*(9*c8 - 3*c6 - 5*c4 - c2) + 
          1/4*(-R22 + R11**2 )*(27*c8 + 3*c6 + c4 + c2) + 
          1/4*(R23 - R12*R11)*(45*c8 + 9*c6 + 7*c4 + 3*c2)
        
        if (iters){
          comment = paste(comment, ", using iterations for p-value approximation")
          pLow = 0
          pHigh = 1
          pVal = 0.05
          nIter = 1
          whileDo = TRUE
          
          while (whileDo){
            cCrit = qchisq(1-pVal, df1)
            
            #(re)calculate chi values
            c2 = cCrit**1/(k + 2*1 - 3)
            c4 = c2 * cCrit/(k + 2*2 - 3)
            c6 = c4 * cCrit/(k + 2*3 - 3)
            c8 = c6 * cCrit/(k + 2*4 - 3)
            
            #calculate Jcrit
            Jcrit = cCrit + 1/2*(3*c4+c2)*lamb + 
              1/16*(3*c4 + c2 )**2*(1-(k-3)/cCrit)*lamb**2 + 
              1/2*(3*c4 + c2 )*
              ((8*R23 - 10*R22 + 4*R21 - 6*R12**2 + 8*R12*R11 - 4*R11**2) + 
                 (2*R23 - 4*R22 + 2*R21 - 2*R12**2 + 4*R12*R11 - 2*R11**2)*(c2 - 1) + 
                 1/4*(-R12**2 + 4*R12*R11 - 2*R12*R10 - 4*R11**2 + 4*R11*R10 - R10**2 )*(3*c4 - 2*c2 - 1)) + 
              (R23 - 3*R22 + 3*R21 - R20)*(5*c6 + 2*c4 + c2) + 
              3/16*(R12**2 - 4*R23 + 6*R22 - 4*R21 + R20)*(35*c8 + 15*c6 + 9*c4 + 5*c2) + 
              1/16*(-2*R22**2 + 4*R21 - R20 + 2*R12*R10 - 4*R11*R10 + R10**2)*(9*c8 - 3*c6 - 5*c4 - c2) + 
              1/4*(-R22 + R11**2 )*(27*c8 + 3*c6 + c4 + c2) + 
              1/4*(R23 - R12*R11)*(45*c8 + 9*c6 + 7*c4 + 3*c2)
            
            if (Jcrit < J){
              pHigh = pVal
              pVal = (pLow + pVal)/2}
            else if (Jcrit > J){
              pLow = pVal
              pVal = (pHigh + pVal)/2}
            
            nIter = nIter + 1
            
            if (Jcrit == J | nIter >= 500){
              whileDo = FALSE}
            }
          
          reject = pVal < alpha
          
          out <- data.frame(J, df1, pVal, reject)
          colnames(out)<-c("statistic", "df1", "p-value", "reject H0")}
        
        else{
          reject = J > Jcrit
          
          out <- data.frame(J, df1, Jcrit, reject)
          colnames(out)<-c("statistic", "df1", "J-critical", "reject H0")}
      }
    }
  }
  
  #p-values for F-distribution tests
  if (test=='fisher' | test=='box' | test=='welch' | test=='box' | test=='brown-forsythe' | test=='mehrotra' | test=='hartung-agac-makabi'){
    pVal = pf(Fstat, df1, df2, lower.tail = FALSE)
    reject = pVal < alpha
    
    out <- data.frame(Fstat, df1, df2, pVal, reject)
    colnames(out)<-c("statistic", "df1", "df2", "p-value", "reject H0")}
  
  #p-value for chi-square distribution tests
  if (test=='cochran' | test=='scott-smith' | test=='alexander-govern' | test=='ozdemir-kurt'){
    pVal = pchisq(chi2Stat, df1, lower.tail=FALSE)
    reject = pVal < alpha
    
    out <- data.frame(chi2Stat, df1, pVal, reject)
    colnames(out)<-c("statistic", "df1", "p-value", "reject H0")}
  
  #add test name
  out = cbind(out, unname(names[test]))
  colnames(out)[length(colnames(out))] <- "test"
  
  #add comment
  if (!is.null(comment)){
    out = cbind(out, comment)
    colnames(out)[length(colnames(out))] <- "comment"
  }
  
  return (out)
  
}
