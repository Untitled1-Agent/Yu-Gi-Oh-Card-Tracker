"""Offline regression tests for bounded, non-blocking image batch downloads."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mock_imports import import_with_module_mocks


class FakeSession:
    def __init__(self):
        self.closed = False
        self.get = Mock(side_effect=AssertionError("Unexpected HTTP request"))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True


@pytest.fixture
def batch_env(tmp_path, monkeypatch):
    # ImageManager creates a global instance on import. Keep even those files
    # under tmp_path, and restore module mocks using the repository helper.
    monkeypatch.chdir(tmp_path)
    session = FakeSession()
    session_factory = Mock(return_value=session)
    module = import_with_module_mocks(
        "src.services.image_manager",
        {
            "nicegui": SimpleNamespace(run=SimpleNamespace(io_bound=asyncio.to_thread)),
            "aiohttp": SimpleNamespace(ClientSession=session_factory),
        },
    )
    manager = module.ImageManager(str(tmp_path / "images"))
    return manager, session, session_factory


def test_empty_batch_reports_completion_without_opening_session(batch_env):
    manager, _, session_factory = batch_env
    progress = []
    asyncio.run(manager.download_batch({}, progress_callback=progress.append))
    assert progress == [1.0]
    session_factory.assert_not_called()


@pytest.mark.parametrize("high_res", [False, True])
def test_cached_batch_reports_completion_without_opening_session(batch_env, high_res):
    manager, _, session_factory = batch_env
    for card_id in (1, 2):
        Path(manager.get_local_path(card_id, high_res)).touch()
    progress = []
    asyncio.run(manager.download_batch(
        {1: "one", 2: "two"}, high_res=high_res, progress_callback=progress.append,
    ))
    assert progress == [1.0]
    session_factory.assert_not_called()


@pytest.mark.parametrize("high_res", [False, True])
def test_mixed_cache_preserves_paths_urls_and_progress(batch_env, monkeypatch, high_res):
    manager, session, session_factory = batch_env
    Path(manager.get_local_path(1, high_res)).touch()
    # An image at the other resolution must not suppress the requested one.
    Path(manager.get_local_path(2, not high_res)).touch()
    downloads = []
    progress = []

    async def download(shared_session, card_id, url, path):
        assert shared_session is session
        downloads.append((card_id, url, path))
        await asyncio.sleep(0)
        return path

    monkeypatch.setattr(manager, "_download_with_session", download)
    urls = {1: "one", 2: "two", 3: "three"}
    asyncio.run(manager.download_batch(
        urls, concurrency=2, high_res=high_res, progress_callback=progress.append,
    ))
    assert sorted(downloads) == [
        (2, "two", manager.get_local_path(2, high_res)),
        (3, "three", manager.get_local_path(3, high_res)),
    ]
    assert urls == {1: "one", 2: "two", 3: "three"}
    assert progress == [0.5, 1.0]
    session_factory.assert_called_once_with()
    assert session.closed


@pytest.mark.parametrize("count,concurrency", [(25, 1), (25, 4), (2, 100), (15000, 20)])
def test_task_count_and_active_downloads_are_bounded(batch_env, monkeypatch, count, concurrency):
    manager, session, _ = batch_env
    monkeypatch.setattr(manager, "image_exists", lambda *args: False)

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        seen = []
        active = 0
        peak_active = 0
        worker_count = min(count, concurrency)
        tasks_before = asyncio.all_tasks()

        async def download(shared_session, card_id, url, path):
            nonlocal active, peak_active
            assert shared_session is session
            active += 1
            peak_active = max(peak_active, active)
            seen.append(card_id)
            if active == worker_count:
                started.set()
            try:
                await release.wait()
                await asyncio.sleep(0)
                return path
            finally:
                active -= 1

        monkeypatch.setattr(manager, "_download_with_session", download)
        progress = []
        batch = asyncio.create_task(manager.download_batch(
            {i: str(i) for i in range(count)}, concurrency=concurrency,
            progress_callback=progress.append,
        ))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            # Count all scheduled tasks, not just active HTTP requests. A
            # semaphore around one task per image does NOT pass this check.
            workers = asyncio.all_tasks() - tasks_before - {batch}
            assert len(workers) == worker_count
        finally:
            release.set()
            await batch

        assert sorted(seen) == list(range(count))
        assert peak_active == worker_count
        assert active == 0
        assert progress == [i / count for i in range(1, count + 1)]
        assert not (asyncio.all_tasks() - tasks_before)

    asyncio.run(scenario())
    assert session.closed


@pytest.mark.parametrize("concurrency", [0, -1])
@pytest.mark.parametrize("urls", [{}, {1: "one"}])
def test_non_positive_concurrency_is_rejected(batch_env, concurrency, urls):
    manager, _, session_factory = batch_env
    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(manager.download_batch(urls, concurrency=concurrency))
    session_factory.assert_not_called()


def test_cache_checks_run_off_loop_but_callbacks_stay_on_loop(batch_env, monkeypatch):
    manager, _, _ = batch_env
    loop_thread = threading.get_ident()
    cache_threads = []
    callback_threads = []
    progress = []

    def image_exists(card_id, high_res):
        cache_threads.append(threading.get_ident())
        return False

    async def download(session, card_id, url, path):
        return path

    def on_progress(value):
        callback_threads.append(threading.get_ident())
        progress.append(value)

    monkeypatch.setattr(manager, "image_exists", image_exists)
    monkeypatch.setattr(manager, "_download_with_session", download)
    asyncio.run(manager.download_batch({1: "one", 2: "two"}, progress_callback=on_progress))
    assert len(cache_threads) == 2
    assert all(thread != loop_thread for thread in cache_threads)
    assert callback_threads == [loop_thread, loop_thread]
    assert progress == [0.5, 1.0]


def test_caller_map_is_snapshotted_before_thread_handoff(batch_env, monkeypatch):
    manager, _, _ = batch_env

    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        urls = {1: "one", 2: "two"}
        seen = []
        original_filter = manager._get_missing_images

        def filter_images(snapshot, high_res):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(timeout=5)
            return original_filter(snapshot, high_res)

        async def download(session, card_id, url, path):
            seen.append((card_id, url))
            return path

        monkeypatch.setattr(manager, "_get_missing_images", filter_images)
        monkeypatch.setattr(manager, "_download_with_session", download)
        batch = asyncio.create_task(manager.download_batch(urls))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            urls.clear()
            urls[3] = "changed"
        finally:
            release.set()
            await batch
        assert sorted(seen) == [(1, "one"), (2, "two")]

    asyncio.run(scenario())


class FakeResponse:
    def __init__(self, status):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def read(self):
        return b"image bytes"


def test_http_failures_still_complete_progress_and_successes_write_files(batch_env):
    manager, session, _ = batch_env

    def get(url):
        if url == "network-error":
            raise OSError("connection failed")
        return FakeResponse(404 if url == "missing" else 200)

    session.get.side_effect = get
    progress = []
    asyncio.run(manager.download_batch(
        {1: "success", 2: "missing", 3: "network-error"},
        progress_callback=progress.append,
    ))
    assert Path(manager.get_local_path(1)).read_bytes() == b"image bytes"
    assert not Path(manager.get_local_path(2)).exists()
    assert not Path(manager.get_local_path(3)).exists()
    assert progress == [1 / 3, 2 / 3, 1.0]
    assert session.get.call_count == 3
    assert session.closed


def test_write_failure_does_not_abort_the_batch(batch_env, monkeypatch):
    manager, session, _ = batch_env
    session.get.side_effect = lambda url: FakeResponse(200)
    original_write = manager._write_file

    def write(path, data):
        if path == manager.get_local_path(1):
            raise OSError("disk full")
        original_write(path, data)

    monkeypatch.setattr(manager, "_write_file", write)
    progress = []
    asyncio.run(manager.download_batch({1: "one", 2: "two"}, progress_callback=progress.append))
    assert not Path(manager.get_local_path(1)).exists()
    assert Path(manager.get_local_path(2)).read_bytes() == b"image bytes"
    assert progress == [0.5, 1.0]
    assert session.closed


def test_cancellation_drains_workers_before_closing_session(batch_env, monkeypatch):
    manager, session, _ = batch_env

    async def scenario():
        started = asyncio.Event()
        never = asyncio.Event()
        active = 0
        cleanup_session_states = []
        tasks_before = asyncio.all_tasks()

        async def download(shared_session, card_id, url, path):
            nonlocal active
            active += 1
            if active == 3:
                started.set()
            try:
                await never.wait()
            finally:
                cleanup_session_states.append(shared_session.closed)
                active -= 1

        monkeypatch.setattr(manager, "_download_with_session", download)
        progress = []
        batch = asyncio.create_task(manager.download_batch(
            {i: str(i) for i in range(50)}, concurrency=3, progress_callback=progress.append,
        ))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
        finally:
            batch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await batch
        assert active == 0
        assert cleanup_session_states == [False, False, False]
        assert progress == []
        assert not (asyncio.all_tasks() - tasks_before)

    asyncio.run(scenario())
    assert session.closed


@pytest.mark.parametrize("error_source", ["worker", "callback"])
def test_unexpected_errors_drain_sibling_workers(batch_env, monkeypatch, error_source):
    manager, session, _ = batch_env

    async def scenario():
        all_started = asyncio.Event()
        never = asyncio.Event()
        active = 0
        cleanup_session_states = []
        tasks_before = asyncio.all_tasks()

        async def download(shared_session, card_id, url, path):
            nonlocal active
            active += 1
            if active == 3:
                all_started.set()
            try:
                if card_id == 0:
                    await all_started.wait()
                    if error_source == "worker":
                        raise RuntimeError("worker failed")
                    return path
                await never.wait()
            finally:
                cleanup_session_states.append(shared_session.closed)
                active -= 1

        def on_progress(value):
            raise RuntimeError("callback failed")

        monkeypatch.setattr(manager, "_download_with_session", download)
        with pytest.raises(RuntimeError, match=f"{error_source} failed"):
            await asyncio.wait_for(manager.download_batch(
                {i: str(i) for i in range(50)}, concurrency=3, progress_callback=on_progress,
            ), timeout=5)
        assert active == 0
        assert cleanup_session_states == [False, False, False]
        assert not (asyncio.all_tasks() - tasks_before)

    asyncio.run(scenario())
    assert session.closed
