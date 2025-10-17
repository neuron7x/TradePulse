// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface ILiquidityManagementHook {
    function beforeAddLiquidity(address sender, bytes calldata data) external;

    function afterAddLiquidity(
        address sender,
        uint256 tokenId,
        uint128 liquidity,
        bytes calldata data
    ) external;

    function afterSwap(
        address sender,
        int256 amount0,
        int256 amount1,
        bytes calldata data
    ) external;
}

