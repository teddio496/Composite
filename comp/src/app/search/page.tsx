"use client"
import { useState, ChangeEvent } from "react";
import { useRouter } from "next/router";

const Search=()=>{
    const [search, setSearch] = useState('');
    const [list, setList] = useState([]);
    const [history, setHistory] = useState(['teddioo','kasumi']);
    const [historyOn, setHistoryOn] = useState(false);
    const historyList=()=>{
        if (!history) return <div>No history</div>
        else return <ul>{history.map((name, index) => <li key={index}>{name}</li>)}</ul>
    }

    return(
        <div>
        <input 
        onFocus={()=> setHistoryOn(true)}
        onBlur={()=> setHistoryOn(false)}
        onChange={(e)=>setSearch(e.target.value)}>
        
        </input>
        <div>{historyOn && historyList()}</div>
        </div>
    )
}

export default Search;