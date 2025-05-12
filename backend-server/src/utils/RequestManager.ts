import { TimeWindow, WindowState } from "../types/request";

export class RequestManager {
    private windows: TimeWindow[];
    private states: WindowState[];
    private queue: (() => Promise<any>)[];
    private running = false;
    
    constructor (windows: TimeWindow[]) {
        this.windows = windows;
        this.states = windows.map(() => ({"count": 0, "startTime": Date.now()}))
        this.queue = [];
    }

    private canRequestNow(): boolean {
        const current_time = Date.now()

        return this.windows.every((window, i) => {
            if (this.states[i].startTime + window.intervalMs < current_time) {
                this.states[i].startTime = current_time;
                this.states[i].count = 0;
                return true;
            }
            return this.states[i].count <= window.limit
        });
    }
    
    private getWaitTime(after: number = 0): number {
        let wait = 0;
        this.windows.forEach((window, i) => {
            if (this.states[i].count >= window.limit) {
                const current_time = Date.now()
                const remaining_time = Math.floor((this.queue.length + after)/window.limit)
                    * window.intervalMs - (current_time - this.states[i].startTime);
                wait = Math.max(wait, remaining_time)
            }  
        });
        return wait;
    }

    private isAllowed(now: number): boolean {
        return this.windows.every((window, i) => {
            const s = this.states[i];
            if (now - s.startTime > window.intervalMs) {
                s.count = 0;
                s.startTime = now;
            }
            return s.count < window.limit;
        });
      }

      private runQueue() {
        if (this.running) return;
        this.running = true;
    
        const tryRun = () => {
            const now = Date.now();       
            if (this.queue.length === 0) {
                this.running = false;
                return;
            }
    
            if (this.isAllowed(now)) {
                this.windows.forEach((window, i) => {
                    const s = this.states[i];
                    if (now - s.startTime > window.intervalMs) {
                        s.count = 0;
                        s.startTime = now;
                    }
                    s.count++;
                });
    
                const job = this.queue.shift();
                if (job) job();
                this.running = false;
                this.runQueue();  // Process next if available

            } else {
                const wait = this.getWaitTime() + 500; // extra time for time delay
                setTimeout(() => {
                    this.running = false;
                    this.runQueue();  // Try again after wait
                }, wait);
            }
        };
    
        tryRun();
    }
    

    public async enqueue<T>(task: () => Promise<T>): Promise<{data?: T}> {
        return new Promise((resolve) => {
            this.queue.push(async () => {
                const result = await task();
                resolve({ data: result });
            });
            this.runQueue();
        });
    }
}