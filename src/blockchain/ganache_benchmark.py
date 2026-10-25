"""Minimal Ganache-backed benchmark for local Ethereum-style gas and latency measurements.

This package intentionally measures a local development chain only. It records the
actual gas used by a compact smart contract in a Ganache environment and reports
transaction latency, throughput and fee estimates. The figures are not mainnet or
real-world energy figures.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import statistics
import subprocess
import time
from pathlib import Path

from solcx import compile_standard, install_solc
from web3 import Web3

DEFAULT_GANACHE_PORT = 8545
DEFAULT_GANACHE_URL = f"http://127.0.0.1:{DEFAULT_GANACHE_PORT}"

SOLIDITY_SOURCE = """
pragma solidity ^0.8.20;

contract FLRoundRegistry {
    struct RoundRecord {
        bytes32 experimentId;
        uint256 roundNumber;
        bytes32 modelHash;
        string status;
    }

    mapping(bytes32 => RoundRecord) private records;
    uint256 public count;

    event RoundCommitted(bytes32 indexed experimentId, uint256 indexed roundNumber, bytes32 modelHash, string status);

    function commit(bytes32 experimentId, uint256 roundNumber, bytes32 modelHash, string calldata status) external {
        records[keccak256(abi.encodePacked(experimentId, roundNumber))] = RoundRecord({
            experimentId: experimentId,
            roundNumber: roundNumber,
            modelHash: modelHash,
            status: status
        });
        count += 1;
        emit RoundCommitted(experimentId, roundNumber, modelHash, status);
    }
}
"""


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def _wait_for_rpc(url: str, timeout_seconds: float = 30.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            web3 = Web3(Web3.HTTPProvider(url))
            if web3.is_connected():
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def _start_ganache(port: int = DEFAULT_GANACHE_PORT, host: str = "127.0.0.1"):
    if _port_open(host, port):
        return None
    npx = shutil.which("npx.cmd") or shutil.which("npx") or "npx.cmd"
    command = [
        npx,
        "--yes",
        "ganache",
        "--wallet.deterministic",
        "--chain.chainId",
        "1337",
        "--server.host",
        host,
        "--server.port",
        str(port),
        "--miner.blockTime",
        "0",
        "--defaultBalanceEther",
        "1000000",
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _wait_for_rpc(f"http://{host}:{port}", timeout_seconds=45.0):
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            pass
        raise TimeoutError(f"Ganache did not start on http://{host}:{port}")
    return process


def _compile_contract() -> tuple[list[dict], str]:
    install_solc("0.8.20")
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {"FLRoundRegistry.sol": {"content": SOLIDITY_SOURCE}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "evm.bytecode.object"]
                    }
                }
            },
        },
        solc_version="0.8.20",
    )
    contract_data = compiled["contracts"]["FLRoundRegistry.sol"]["FLRoundRegistry"]
    return contract_data["abi"], contract_data["evm"]["bytecode"]["object"]


def _ensure_tx_receipt(w3: Web3, tx_hash, timeout_seconds: float = 60.0):
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            receipt = w3.eth.get_transaction_receipt(tx_hash)
            if receipt is not None:
                return receipt
        except Exception:
            pass
        time.sleep(0.05)
    raise TimeoutError(f"Timed out awaiting transaction {tx_hash.hex() if isinstance(tx_hash, (bytes, bytearray)) else tx_hash}")


def run_ganache_benchmark(
    url: str = DEFAULT_GANACHE_URL,
    output_root: str | Path = "results/blockchain/ganache",
    rounds: int = 20,
    warmup: int = 5,
    port: int = DEFAULT_GANACHE_PORT,
    host: str = "127.0.0.1",
):
    """Run a compact Ganache benchmark and write a JSON summary to disk."""
    process = _start_ganache(port=port, host=host)
    started = process is not None
    try:
        web3 = Web3(Web3.HTTPProvider(url))
        if not web3.is_connected():
            raise ConnectionError(f"Could not connect to local Ganache at {url}")

        abi, bytecode = _compile_contract()
        account = web3.eth.accounts[0]
        gas_price_wei = int(web3.eth.gas_price)

        contract = web3.eth.contract(abi=abi, bytecode=bytecode)
        deploy_tx = contract.constructor().transact({"from": account, "gas": 3_000_000, "gasPrice": gas_price_wei})
        deploy_receipt = web3.eth.wait_for_transaction_receipt(deploy_tx, timeout=60)
        contract_instance = web3.eth.contract(address=deploy_receipt["contractAddress"], abi=abi)

        samples = []
        for round_index in range(rounds + warmup):
            experiment_id = Web3.keccak(text=f"experiment-{round_index}")
            model_hash = Web3.keccak(text=f"model-{round_index}")
            status = f"completed_{round_index:02d}"
            nonce = web3.eth.get_transaction_count(account)
            start_time = time.perf_counter()
            tx_hash = contract_instance.functions.commit(experiment_id, round_index, model_hash, status).transact(
                {"from": account, "gasPrice": gas_price_wei, "gas": 220_000, "nonce": nonce}
            )
            receipt = _ensure_tx_receipt(web3, tx_hash, timeout_seconds=60.0)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if round_index < warmup:
                continue
            samples.append(
                {
                    "round": round_index - warmup,
                    "transaction_hash": receipt["transactionHash"].hex(),
                    "gas_used": receipt["gasUsed"],
                    "gas_price_wei": gas_price_wei,
                    "fee_wei": int(receipt["gasUsed"]) * int(gas_price_wei),
                    "fee_eth": float(Web3.from_wei(int(receipt["gasUsed"]) * int(gas_price_wei), "ether")),
                    "latency_ms": round(elapsed_ms, 3),
                    "block_number": receipt["blockNumber"],
                    "status": status,
                }
            )

        if not samples:
            raise ValueError("No transaction samples collected from Ganache benchmark")

        gas_values = [row["gas_used"] for row in samples]
        latency_values = [row["latency_ms"] for row in samples]
        fee_values = [row["fee_eth"] for row in samples]
        throughput_tx_per_s = len(samples) / max(sum(row["latency_ms"] for row in samples) / 1000.0, 1e-9)
        summary = {
            "rpc_url": url,
            "ganache_started": started,
            "ganache_version": "ganache v7.9.2",
            "network_id": 1337,
            "gas_price_wei": gas_price_wei,
            "gas_price_gwei": float(Web3.from_wei(gas_price_wei, "gwei")),
            "tx_count": len(samples),
            "mean_gas_used": float(statistics.mean(gas_values)),
            "median_gas_used": float(statistics.median(gas_values)),
            "max_gas_used": max(gas_values),
            "mean_latency_ms": float(statistics.mean(latency_values)),
            "median_latency_ms": float(statistics.median(latency_values)),
            "p95_latency_ms": float(sorted(latency_values)[max(0, int(0.95 * len(latency_values)) - 1)]),
            "mean_fee_eth": float(statistics.mean(fee_values)),
            "median_fee_eth": float(statistics.median(fee_values)),
            "throughput_tx_per_s": float(throughput_tx_per_s),
            "methodology": "Actual local Ganache benchmark; gas and fee values are in the simulated chain unit. No mainnet or real-world energy figures are claimed.",
        }

        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "rounds": rounds,
                "warmup": warmup,
                "port": port,
                "host": host,
                "contract": "FLRoundRegistry",
                "rpc_url": url,
            },
            "samples": samples,
            "summary": summary,
            "notes": [
                "Transaction gas is recorded from the actual receipt on the local Ganache chain.",
                "Fee estimates use the chain's current gas price and convert wei to ether.",
                "Energy use is not directly measured because Ganache does not expose a production energy model.",
            ],
        }
        (output_path / "ganache_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
    finally:
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                pass


if __name__ == "__main__":
    run_ganache_benchmark()
