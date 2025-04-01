import { useState, ChangeEvent } from "react";
import { useRouter } from "next/router";

export const Search=()=>{
    const [search, setSearch] = useState('');
    const [list, setList] = useState([]);


    return(
        <div>Search</div>
    )
}