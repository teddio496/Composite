"use client"
import { useState, ChangeEvent, useEffect, SetStateAction } from "react";
import { useRouter } from "next/navigation";

const Search=()=>{
    const [search, setSearch] = useState('');
    const [history, setHistory] = useState<String []>([]);
    const [historyOn, setHistoryOn] = useState(false);
    const router = useRouter();

    //This function get the value(string) from the local storage, and add into the searchhistory list.
    const addSearchHistory=()=>{  
        const a = JSON.parse(localStorage.getItem("riotID")||"[]");
        setHistory(a)
    }

    const historyList=()=>{
        console.log(history)
        console.log(!history)
        if (history.length === 0) return <div>No history</div>
        else return <ul>{history.map((name, index) => <li key={index}>{name}</li>)}</ul>
    }
    //adding entered value into localstorage
    const handleSubmit = (value: String) => {
        if (history.includes(value)) {
            return;
        }
        const updatedHistory = [...history, value];
        localStorage.setItem("riotID", JSON.stringify(updatedHistory));
        setHistory(updatedHistory);
        router.push(`/profile/${value}`)
    }

    useEffect(() => {
        addSearchHistory();
    },[])
    

    //This is a function for triggering to search by press enter key
    const handleKeyPress = (e : any) => {
        console.log(e)
        if (e.key == 'Enter'){
            console.log("perfect")
            console.log(e.key)
            handleSubmit(e.target.value)
        }
    };


    return(
        <div>
        <input 
        onFocus={()=> setHistoryOn(true)}
        onBlur={()=> setHistoryOn(false)}
        onChange={(e)=>setSearch(e.target.value)}
        onKeyDown={handleKeyPress}>
        
        </input>
        <div>{historyOn && historyList()}</div>
        </div>
    )
}

export default Search;