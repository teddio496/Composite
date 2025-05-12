import { RequestManager } from '../../src/utils/RequestManager';

jest.useFakeTimers();

describe('RequestManager', () => {
  let manager: RequestManager;

  beforeEach(() => {
    manager = new RequestManager([
      { limit: 2, intervalMs: 1000 },  // Lower limit for easier testing
      { limit: 4, intervalMs: 5000 },
    ]);
  });

  test('executes task immediately if under limit', async () => {
    const task = jest.fn(() => Promise.resolve('success'));

    const p = manager.enqueue(task);
    await p;

    expect(task).toHaveBeenCalledTimes(1);
  });

  test('queues task when over limit and processes later', async () => {
    const results: string[] = [];

    // Add 2 tasks to hit the 2-per-second limit
    manager.enqueue(() => Promise.resolve(results.push('task1')));
    manager.enqueue(() => Promise.resolve(results.push('task2')));

    // This one should queue (since limit=2)
    const thirdTask = manager.enqueue(() => Promise.resolve(results.push('task3')));

    // Fast-forward 1s to allow window to reset
    jest.advanceTimersByTime(1000);
    await thirdTask;

    expect(results).toEqual(['task1', 'task2', 'task3']);
  });

  test('queues multiple tasks and respects order', async () => {
    const results: number[] = [];

    // Enqueue 5 tasks, limit is 2/sec and 4/5sec
    const promises = [];
    for (let i = 1; i <= 5; i++) {
      promises.push(manager.enqueue(() => Promise.resolve(results.push(i))));
    }

    // 2 tasks should run immediately
    await Promise.resolve(); // flush microtasks
    expect(results).toEqual([1, 2]);

    // Fast-forward 1s -> next 2 should run (respects 2/sec limit)
    jest.advanceTimersByTime(1000);
    await Promise.resolve();
    expect(results).toEqual([1, 2, 3, 4]);

    // Fast-forward 1s -> last task should run
    jest.advanceTimersByTime(1000);
    await Promise.all(promises);

    expect(results).toEqual([1, 2, 3, 4, 5]);
  });

  test('respects longer window (e.g., 5s limit)', async () => {
    const results: number[] = [];

    // Fill up 4 requests (limit for 5s window)
    const promises = [];
    for (let i = 1; i <= 4; i++) {
      promises.push(manager.enqueue(() => Promise.resolve(results.push(i))));
    }

    await Promise.all(promises);
    expect(results).toEqual([1, 2, 3, 4]);

    // Add a 5th request: should NOT run yet (5s window hit)
    const fifth = manager.enqueue(() => Promise.resolve(results.push(5)));

    // Fast-forward 1s: shouldn't run yet
    jest.advanceTimersByTime(1000);
    await Promise.resolve();
    expect(results).toEqual([1, 2, 3, 4]);

    // Fast-forward 5s: should now run
    jest.advanceTimersByTime(5000);
    await fifth;
    expect(results).toEqual([1, 2, 3, 4, 5]);
  });

  test('does not exceed limits under heavy load', async () => {
    const task = jest.fn(() => Promise.resolve('ok'));

    // Enqueue 10 tasks rapidly
    const promises = [];
    for (let i = 0; i < 10; i++) {
      promises.push(manager.enqueue(task));
    }

    // Only 2 should run now (2/sec limit)
    await Promise.resolve();
    expect(task).toHaveBeenCalledTimes(2);

    // Fast-forward 1s at a time to let rest process
    for (let i = 0; i < 5; i++) {
      jest.advanceTimersByTime(1000);
      await Promise.resolve();
    }

    await Promise.all(promises);
    expect(task).toHaveBeenCalledTimes(10);
  });
});
