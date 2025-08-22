# Urban-carbon-emissions-model-bridging-dynamic-drivers-for-cross-scale-policy-insights
Model to analyze urban carbon emissions. If your data come across the charateristics that in different stages, new variables may be brought in, then, this model is for you.
**Problem Statement:**
- When it comes to analyze urban carbon emisisons, GDP, energy consumption and industrial structure are the most commonly factors to be considered in the research. However, other factors may also counts. We collected 58 factors that may contribute to urban carbon emissions. But we found that in different developing stages, new variables may be brought in.

**Characteristics of My Code:**
- Dummy variables were used to translate the changes of variables in different developing stages.
- Ridge and Lasso regression models were used to deal with multicollinearity among indicators.

**Result of My Code:**
- Take Shenzhen City as an example, both models showed a strong performance. Ridge model got an MSE of 0.074 and R2 of 99.59%. Lasso model recorded an MSE of 2.304 and R2 of 87.33%. Under the outlier test, Ridge regression again outperformed Lasso, with an MSE of 0.735 and R2 of 95.96%, compared to 2.438 and 86.60% for the Lasso model. Under rolling window validation results(2011-2023), both models exhibited low prediction error and stability across windows.
