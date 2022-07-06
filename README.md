# About Dataset
## Context
This classic dataset contains the prices and other attributes of almost 54,000 diamonds.
## Content
- **price** price in US dollars (\$326--\$18,823)
- **carat** weight of the diamond (0.2--5.01)
- **cut** quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **color** diamond colour, from J (worst) to D (best)
- **clarity** a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
- **x** length in mm (0--10.74)
- **y** width in mm (0--58.9)
- **z** depth in mm (0--31.8)
- **depth** total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
- **table** width of top of diamond relative to widest point (43--95)
## Importing Libraries
![image](https://user-images.githubusercontent.com/56602084/177603825-536540cb-1b5a-42aa-b0ba-45e056b32933.png)
## Loading Data
![image](https://user-images.githubusercontent.com/56602084/177604243-6e9cd9d9-a7b7-4c32-ba14-f5002232c8f2.png)
![image](https://user-images.githubusercontent.com/56602084/177604383-4262b712-ae6e-4aba-ba45-c3d48fb6cf90.png)
## Data Preprocessing
**Step involved in Data Preprocessing**
- Data Cleaning
- Identifying and removing outliers
- Encoding categorical variables

![image](https://user-images.githubusercontent.com/56602084/177604478-6f137a18-8a22-4591-8f06-0fbec53d1556.png)

The first column is an index ("Unnamed: 0") and thus we are going to remove it.
![image](https://user-images.githubusercontent.com/56602084/177604796-280435b8-7542-4064-9506-3bd8cd5a66d8.png)

Min value of "x", "y", "z" are zero this indicates that there are faulty values in data that represents dimensionless or 2-dimensional diamonds. So we need to filter out those as it clearly faulty data points.

![image](https://user-images.githubusercontent.com/56602084/177604929-6761e264-30d4-422d-97f6-9edc17a835bd.png)

We lost 20 data points by deleting the dimensionless(2-D or 1-D) diamonds.

**Checking for null values**

![image](https://user-images.githubusercontent.com/56602084/177605525-716e6502-e1df-436d-8f93-59c5b45a4959.png)

we can see that the data is  cleaned

# The rest of this project you can see in the linked jupyter notebook file


