-- No import needed
CREATE PROCEDURE GreetUser @Name NVARCHAR(50)
AS
BEGIN
    SELECT 'Hello, ' + @Name + '!';
END;

