select*from houses

--1. Average price grouped by city
SELECT city,
       AVG(price) AS average_price
FROM houses
GROUP BY city;

--2. Cities where average price is above the overall average
SELECT city, AVG(price)
FROM houses
GROUP BY city
HAVING AVG(price) > (
    SELECT AVG(price)
    FROM houses
);

--3. Count of houses grouped by bedrooms
SELECT bedrooms, COUNT(*)
FROM houses
GROUP BY bedrooms;

--4. Average price for houses with bathrooms > 2
SELECT bathrooms, AVG(price)
FROM houses
WHERE bathrooms > 2
GROUP BY bathrooms;

--5. Houses priced above the overall average (Subquery)
SELECT *
FROM houses
WHERE price > (
    SELECT AVG(price)
    FROM houses
);