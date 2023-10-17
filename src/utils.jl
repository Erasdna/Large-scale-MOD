function cycle_and_replace!(mat,new_element; col=true)
    """
        shifts rows or columns one and replaces the last element
    """
    if col
        mat .= circshift(mat,(0,-1))
        mat[:,end] .= new_element
    else 
        mat .= circshift(mat,(-1,0))
        mat[end,:] .= new_element
    end
end