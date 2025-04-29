using CSV
using DataFrames 


function load_data(dataname) 
    df = "./data/$(dataname).csv" |> CSV.File |> DataFrame
    
    df_ = nothing
    if dataname == "faithful_data.csv"
        df_ = df[!, 2:3]
    elseif dataname == "GvHD"
        df_ = df[df.CD3 .> 300, :]
        df_ = df_[!, [:CD8, :CD4]] 
    else 
        df_ = df[!, 1:2]
    end  

    return df_ |> Matrix |> transpose 
end  
