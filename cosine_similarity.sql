-- Function to calculate cosine similarity between two JSONB arrays
CREATE OR REPLACE FUNCTION cosine_similarity(a JSONB, b JSONB)
RETURNS FLOAT AS $$
DECLARE
    dot_product FLOAT := 0;
    norm_a FLOAT := 0;
    norm_b FLOAT := 0;
    i INTEGER;
    val_a FLOAT;
    val_b FLOAT;
    len_a INTEGER;
    len_b INTEGER;
BEGIN
    -- Check if both arrays have the same length
    len_a := jsonb_array_length(a);
    len_b := jsonb_array_length(b);
    
    IF len_a != len_b OR len_a = 0 THEN
        RETURN 0;
    END IF;
    
    -- Calculate dot product and norms
    FOR i IN 0..len_a-1 LOOP
        val_a := (a->>i)::FLOAT;
        val_b := (b->>i)::FLOAT;
        
        dot_product := dot_product + (val_a * val_b);
        norm_a := norm_a + (val_a * val_a);
        norm_b := norm_b + (val_b * val_b);
    END LOOP;
    
    -- Avoid division by zero
    IF norm_a = 0 OR norm_b = 0 THEN
        RETURN 0;
    END IF;
    
    -- Return cosine similarity
    RETURN dot_product / (sqrt(norm_a) * sqrt(norm_b));
END;
$$ LANGUAGE plpgsql IMMUTABLE;